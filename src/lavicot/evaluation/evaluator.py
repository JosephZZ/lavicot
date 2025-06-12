import os
import sys
import argparse
import json
import torch
import wandb
import random
from typing import List, Dict, Any, Optional, Union, Tuple, TypeVar
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import re

from ..models.lavicot_bias import (
    add_instance_level_prefix_generator,
    create_test_time_prefix_config,
    TestTimePrefixModel
)
from ..config.config_loader import load_config, save_config
from ..utils.logging_utils import (
    count_parameters, get_gpu_memory_usage
)
from ..utils.data_utils import get_formatted_data, get_data_extractor
from ..utils.tokenizer_utils import get_generation_pad_token_id

def extract_reasoning_from_output(generated_text: str) -> str:
    """Extract reasoning from generated output that contains ChatML format."""
    assistant_part = generated_text.split("<|im_start|>assistant")[1]
    if "<|im_end|>" in assistant_part:
        assistant_part = assistant_part.split("<|im_end|>")[0]
    
    # Look for thinking tags in the assistant's response
    if "<thinking>" in assistant_part  in assistant_part:
        reasoning = assistant_part.split("<thinking>")[1].split("</thinking>")[0]
        return reasoning.strip()
    
    # If no thinking tags, return the whole assistant part
    return assistant_part.strip()


def extract_reasoning_and_answer_from_generated_text(generated_text: str, extract_number_only: bool = True) -> Tuple[str, str]:
    """Extract reasoning and answer from generated text in ChatML format."""
    # Check if assistant section exists
    if "<|im_start|>assistant" not in generated_text:
        return "", ""
    
    # Find assistant section
    assistant_section = generated_text.split("<|im_start|>assistant")[1]
    if "<|im_end|>" in assistant_section:
        assistant_section = assistant_section.split("<|im_end|>")[0]
    
    # Extract reasoning from thinking tags
    reasoning = ""
    if "<thinking>" in assistant_section in assistant_section:
        reasoning = assistant_section.split("<thinking>")[1].split("</thinking>")[0].strip()
    
    # Extract answer from answer tags
    answer = ""
    if "<answer>" in assistant_section in assistant_section:
        answer = assistant_section.split("<answer>")[1].split("</answer>")[0].strip()
    
    if extract_number_only:
        answer = re.search(r'\d+', answer).group()
    return reasoning, answer

T = TypeVar('T')

# Import shared model utilities 
from ..models.base_model_integration import setup_model_and_tokenizer, setup_prefix_generator

def setup_model_and_tokenizer_for_eval(
    model_name: str,
    device: str,
    config_dict: dict,
    initial_prefixes: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None
) -> Tuple[TestTimePrefixModel, AutoTokenizer]:
    """Initialize model and tokenizer for evaluation with proper configuration."""
    print(f"Loading model for evaluation: {model_name}")
    
    # Use shared setup functions
    base_model, tokenizer = setup_model_and_tokenizer(model_name, device)
    model = setup_prefix_generator(
        base_model, 
        device,
        config_dict,
        tokenizer,
        initial_prefixes,
        initial_states
    )
    
    return model, tokenizer

def evaluate_model_configurations(
    model: TestTimePrefixModel,
    tokenizer: AutoTokenizer,
    eval_instances: List[Dict[str, Any]],
    dataset_name: str,
    device: str,
    max_length: int,
    evaluation_settings: List[Dict[str, Any]],
    temperature: float = 0.7
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, List[Dict]]]:
    """Evaluate model with different configurations.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for the model
        eval_instances: List of data instances (e.g., from dataset)
        data_extractor: Function to extract (question, reasoning, answer) from each instance
        device: Device to run evaluation on
        max_length: Maximum sequence length
        evaluation_settings: List of evaluation configuration dictionaries
        
    Returns:
        Tuple of (results_dict, model_outputs_dict)
    """
    results = {}
    model_outputs = {}  # Store model outputs for each setting
    data_extractor = get_data_extractor(dataset_name)

    # Initialize results structure
    for setting in evaluation_settings:
        setting_index = setting["setting_index"]
        results[f"setting_{setting_index}"] = {"accuracy": 0.0}
        model_outputs[f"setting_{setting_index}"] = []
    
    # Evaluate each instance across all settings
    for q_idx, instance in enumerate(tqdm(eval_instances, desc="Evaluating")):
        # Extract question and answer from the instance
        question, reasoning, ground_truth_answer = data_extractor(instance)
        # Store prefixes and CoTs for this specific question across settings
        stored_prefixes = {}  # Reset for each question
        stored_outputs = {}      # Reset for each question
        
        for setting in evaluation_settings:
            setting_index = setting["setting_index"]
            
            # Get configuration for this setting
            reuse_previous_prefix = setting["reuse_previous_prefix"]
            setting_index_for_prefix_reuse = setting["setting_index_for_prefix_reuse"]
            reuse_previous_cot = setting["reuse_previous_cot"]
            setting_index_for_previous_cot = setting["setting_index_for_previous_cot"]
            proportion_prev_cot_reuse = setting["proportion_prev_cot_reuse"]
            iterations = setting["iterations"]
            try:
                # Reset model prefixes for each example
                model.reset_prefixes()
                
                # Initialize or get previous prefix
                if reuse_previous_prefix and setting_index_for_prefix_reuse is not None and setting_index_for_prefix_reuse in stored_prefixes:
                    # Reuse prefix from previous setting
                    stored_prefix = stored_prefixes[setting_index_for_prefix_reuse]
                    model.current_prefixes = stored_prefix.clone().to(device)
                    model._set_layer_prefixes()  # Apply the prefixes to the attention layers
                
                # Prepare input based on configuration using consistent ChatML format
                if reuse_previous_cot and setting_index_for_previous_cot is not None and setting_index_for_previous_cot in stored_outputs:
                    # Use previous CoT output
                    prev_cot_raw = stored_outputs[setting_index_for_previous_cot]
                    # Extract reasoning from the previous generated output
                    prev_reasoning = extract_reasoning_from_output(prev_cot_raw)
                    
                    if proportion_prev_cot_reuse is not None:
                        # Use partial previous CoT - don't close the thinking tag
                        cot_length = int(len(prev_reasoning) * proportion_prev_cot_reuse)
                        partial_reasoning = prev_reasoning[:cot_length]
                        
                        input_text = get_formatted_data(question, partial_reasoning, None, add_im_end=False, close_thinking=False)
                    else:
                        raise ValueError("proportion_prev_cot_reuse must be provided")
                else:
                    # Use only question - consistent with "question_only" mode in training
                    input_text = get_formatted_data(question, None, None)
                
                # Generate with specified number of prefix iterations
                try:
                    # Tokenize input with proper batch handling
                    tokenized = tokenizer(
                        input_text, 
                        return_tensors="pt", 
                        padding=True,
                        truncation=True,
                        max_length=max_length
                    )
                    
                    input_ids = tokenized.to(device)
                    
                    with torch.no_grad():    
                        # Update prefixes based on current input with specified iterations
                        model.update_prefix_given_input(
                            input_ids=input_ids["input_ids"],
                            num_iterations=iterations  # Pass iterations to prefix generator
                        )
                        # model.set_zero_prefixes()
                        # print("debug: set zero prefixes")
                        # Generate sequence
                        generated = model.base_model.generate(
                            input_ids=input_ids["input_ids"],
                            attention_mask=input_ids.get("attention_mask"),
                            max_new_tokens=max_length,
                            temperature=temperature,
                            top_p=0.95,
                            do_sample=True,
                            pad_token_id=get_generation_pad_token_id(tokenizer),
                            eos_token_id=tokenizer.eos_token_id
                        )
                        
                        # Decode output
                        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
                        
                except Exception as gen_error:
                    print(f"Error during generation: {str(gen_error)}")
                    generated_text = f"Error: {str(gen_error)}"
                
                # Store CoT for potential reuse
                stored_outputs[setting_index] = generated_text
                
                # Store current prefixes for potential reuse by later settings
                if model.current_prefixes is not None:
                    stored_prefixes[setting_index] = model.current_prefixes.clone().detach()
                
                # Extract reasoning and answer
                generated_reasoning, generated_answer = extract_reasoning_and_answer_from_generated_text(generated_text)
                
                # Store complete output for this example
                example_output = {
                    "question": question,
                    "true_reasoning": reasoning,
                    "true_answer": ground_truth_answer,
                    "output": generated_text,
                    "generated_reasoning": generated_reasoning,
                    "generated_answer": generated_answer,
                    "iterations_used": iterations,
                    "is_correct": generated_answer.strip() == ground_truth_answer.strip()
                }

                # Store output for this setting
                model_outputs[f"setting_{setting_index}"].append(example_output)
                
                # Print example output for first few examples per setting
                if q_idx < 3:  # Print first 3 questions for each setting
                    print("====================trained model output======================")
                    print(f"\nQuestion {q_idx + 1} for Setting {setting_index}:")
                    print(f"Question: {question}")
                    print(f"True Answer: {ground_truth_answer}")
                    print(f"Iterations used: {iterations}")
                    print("Model Output:")
                    print(generated_text)
                    print(f"Generated Answer: {generated_answer}")
                    print(f"Correct: {example_output['is_correct']}")

                    print_raw_model_output = True
                    if print_raw_model_output:
                        model._clear_layer_prefixes()
                        print("--------------------raw model output----------------------")
                        generated_raw = model.base_model.generate(
                            input_ids=input_ids["input_ids"],
                            attention_mask=input_ids.get("attention_mask"),
                            max_new_tokens=max_length,
                            temperature=temperature,
                            top_p=0.95,
                            do_sample=True,
                            pad_token_id=get_generation_pad_token_id(tokenizer),
                            eos_token_id=tokenizer.eos_token_id
                        )               
                        # Decode output
                        generated_text_raw = tokenizer.decode(generated_raw[0], skip_special_tokens=True)
                        print(f"debug: generated_text_raw:\n {generated_text_raw}\n\n")
                    print("==========================================")
                
            except Exception as e:
                print(f"Error during evaluation: {str(e)}")
                # Add failed example to outputs
                model_outputs[f"setting_{setting_index}"].append({
                    "question": question,
                    "true_answer": ground_truth_answer,
                    "output": f"Error: {str(e)}",
                    "iterations_used": iterations,
                    "is_correct": False
                })
                continue
    
    # Calculate final accuracies for each setting
    for setting in evaluation_settings:
        setting_index = setting["setting_index"]
        setting_outputs = model_outputs[f"setting_{setting_index}"]
        
        correct = sum(1 for output in setting_outputs if output["is_correct"])
        total = len(setting_outputs)
        accuracy = (correct / total) * 100 if total > 0 else 0
        
        results[f"setting_{setting_index}"] = {"accuracy": accuracy}
        print(f"Setting {setting_index} Final Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    return results, model_outputs


def evaluate_batch_sequence_prediction_loss(
    model: TestTimePrefixModel,
    tokenizer: AutoTokenizer,
    eval_instances: List[Dict[str, Any]],
    data_extractor: callable,
    device: str,
    max_length: int,
    prefix_iterations_range: Tuple[int, int] = (1, 10)
) -> List[Dict[str, Any]]:
    """Evaluate sequence prediction loss for a batch of instances with varying prefix iterations.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for the model
        eval_instances: List of data instances
        data_extractor: Function to extract (question, reasoning, answer) from instance
        device: Device to run evaluation on
        max_length: Maximum sequence length
        prefix_iterations_range: (min_iterations, max_iterations) for prefix generation
        
    Returns:
        List of loss dictionaries, each mapping iteration_count -> loss
    """
    results = []
    min_iter, max_iter = prefix_iterations_range
    
    model.eval()
    for idx, instance in enumerate(tqdm(eval_instances, desc="Evaluating sequence prediction loss")):
        question, reasoning, answer = data_extractor(instance)
        
        # Reset model prefixes once at the start
        model.reset_prefixes()
        
        # Prepare question-only input for prefix generation  
        from ..utils.data_utils import get_formatted_data
        question_input = get_formatted_data(question, None, None)
        
        # Tokenize question input
        question_tokenized = tokenizer(
            question_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)
        
        # Prepare full sequence for loss evaluation
        full_sequence_text = get_formatted_data(question, reasoning, answer)
        full_sequence_tokenized = tokenizer(
            full_sequence_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)
        
        # Record loss for each iteration incrementally
        losses_by_iteration = {}
        
        for num_iterations in range(min_iter, max_iter + 1):
            try:
                # Run one more iteration (or initial iterations for first time)
                iterations_to_run = 1 if num_iterations > min_iter else min_iter
                
                with torch.no_grad():
                    model.update_prefix_given_input(
                        input_ids=question_tokenized["input_ids"],
                        num_iterations=iterations_to_run
                    )
                
                # Compute loss on full sequence with current prefixes
                with torch.no_grad():
                    outputs = model(
                        input_ids=full_sequence_tokenized["input_ids"],
                        attention_mask=full_sequence_tokenized.get("attention_mask"),
                        labels=full_sequence_tokenized["input_ids"]
                    )
                    loss = outputs.loss.item()
                
                # Record loss for this iteration count
                losses_by_iteration[num_iterations] = loss
                    
            except Exception as e:
                print(f"Error evaluating instance {idx} with {num_iterations} iterations: {e}")
                losses_by_iteration[num_iterations] = float('inf')
                continue
        
        results.append(losses_by_iteration)
        
        # Print progress for first few instances
        if idx < 3:
            print(f"Instance {idx}: Losses by iteration: {losses_by_iteration}")
    
    return results

def evaluate(
    model_name: str,
    config_path: Optional[str] = None,
    dataset_config_name: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    num_eval_samples: int = 100,
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    output_dir: Optional[str] = None
):
    """Standalone evaluation function.
    
    Args:
        model_name: Name or path of the model to load
        config_path: Path to the configuration file
        dataset_config_name: Dataset configuration name (e.g., 'gsm8k', 'math')
        checkpoint_path: Path to the checkpoint file containing saved prefixes and states
        num_eval_samples: Number of samples to evaluate
        use_wandb: Whether to use Weights & Biases logging
        wandb_project: W&B project name
        wandb_entity: W&B entity name
        wandb_run_name: W&B run name
        output_dir: Directory to save outputs
    """
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load configuration
    if config_path:
        config = load_config(config_path, dataset_config_name=dataset_config_name)
    else:
        # Use default config from parent directory
        config = load_config(dataset_config_name=dataset_config_name)
    
    # Initialize wandb if enabled
    if use_wandb:
        wandb.init(
            project=wandb_project or config.wandb_project,
            entity=wandb_entity or config.wandb_entity,
            name=wandb_run_name or config.wandb_run_name,
            config=vars(config)  # Convert back to dict for wandb
        )
    
    # Create output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load saved prefixes and states if checkpoint provided
    initial_prefixes = None
    initial_states = None
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            # Extract prefixes and states from model state dict
            state_dict = checkpoint["model_state_dict"]
            prefix_keys = [k for k in state_dict.keys() if "current_prefixes" in k]
            state_keys = [k for k in state_dict.keys() if "current_states" in k]
            
            if prefix_keys:
                initial_prefixes = state_dict[prefix_keys[0]]
                print(f"Loaded initial prefixes with shape: {initial_prefixes.shape}")
            if state_keys:
                initial_states = state_dict[state_keys[0]]
                print(f"Loaded initial states with shape: {initial_states.shape}")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer_for_eval(
        model_name,
        device,
        vars(config),
        initial_prefixes=initial_prefixes,
        initial_states=initial_states
    )
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(config.dataset_name, config.dataset_config)
    test_dataset = dataset["test"]
    
    # Sample evaluation examples - keep as instances
    num_eval_samples = len(test_dataset) if num_eval_samples == 'full' else int(num_eval_samples)
    eval_indices = random.sample(range(len(test_dataset)), num_eval_samples)
    eval_instances = [test_dataset[i] for i in eval_indices]
    
    # Import the appropriate data extraction function
    from ..utils.data_utils import extract_gsm8k_data_components
    data_extractor = extract_gsm8k_data_components  # Default to GSM8K format
    
    print(f"Evaluating on {len(eval_instances)} samples")
    
    # Run evaluation
    print("\nRunning evaluation...")
    model.eval()
    with torch.no_grad():
        eval_results, model_outputs = evaluate_model_configurations(
            model, tokenizer, eval_instances, data_extractor,
            device, config.max_length, config.evaluation_settings
        )
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Model: {model_name}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Number of samples: {num_eval_samples}")
    print("\nResults by setting:")
    for setting_name, metrics in eval_results.items():
        print(f"{setting_name}: {metrics['accuracy']:.2f}%")
    
    # Log to wandb if enabled
    if use_wandb:
        # Log model details
        param_counts = count_parameters(model)
        wandb.log({
            "model/trainable_params": param_counts["trainable"],
            "model/non_trainable_params": param_counts["non_trainable"],
            "model/total_params": param_counts["total"],
            "model/trainable_percentage": (param_counts["trainable"] / param_counts["total"] * 100),
            "model/gpu_memory_mb": get_gpu_memory_usage()
        })
        
        # Log evaluation results
        for setting_name, metrics in eval_results.items():
            wandb.log({
                f"eval/{setting_name}/accuracy": metrics["accuracy"]
            })
        
        # Log example outputs
        for setting_name, outputs in model_outputs.items():
            # Log first 3 examples for each setting
            for i, example in enumerate(outputs[:3]):
                wandb.log({
                    f"examples/{setting_name}/example_{i+1}/question": example["question"],
                    f"examples/{setting_name}/example_{i+1}/true_answer": example["true_answer"],
                    f"examples/{setting_name}/example_{i+1}/is_correct": example["is_correct"]
                })
                # Log iterations
                for iter_out in example["iterations"]:
                    wandb.log({
                        f"examples/{setting_name}/example_{i+1}/iteration_{iter_out['iteration']}": iter_out["output"]
                    })
    
    # Save results
    if output_dir:
        # Save metrics
        results_file = os.path.join(output_dir, "evaluation_results.json")
        with open(results_file, "w") as f:
            json.dump({
                "model": model_name,
                "dataset": config.dataset_name,
                "num_samples": num_eval_samples,
                "results": eval_results,
                "model_details": {
                    "trainable_params": param_counts["trainable"],
                    "non_trainable_params": param_counts["non_trainable"],
                    "total_params": param_counts["total"],
                    "trainable_percentage": (param_counts["trainable"] / param_counts["total"] * 100),
                    "gpu_memory_mb": get_gpu_memory_usage()
                }
            }, f, indent=2)
        
        # Save model outputs
        outputs_file = os.path.join(output_dir, "model_outputs.json")
        with open(outputs_file, "w") as f:
            json.dump({
                "model": model_name,
                "dataset": config.dataset_name,
                "num_samples": num_eval_samples,
                "settings": model_outputs
            }, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
        print(f"Model outputs saved to {outputs_file}")
    
    if use_wandb:
        wandb.finish()

def main():
    """Parse command line arguments and run evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate LaViCoT model")
    parser.add_argument("--model_name", type=str, required=True, help="Name or path of the model to evaluate")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--num_eval_samples", type=int, default=1319, help="Number of samples to evaluate")
    parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, help="Wandb entity name")
    parser.add_argument("--wandb_run_name", type=str, help="Wandb run name")
    parser.add_argument("--output_dir", type=str, help="Directory to save results")
    
    args = parser.parse_args()
    
    evaluate(
        model_name=args.model_name,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        num_eval_samples=args.num_eval_samples,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main() 