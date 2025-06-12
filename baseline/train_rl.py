import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    GenerationConfig
)
from trl import GRPOTrainer, GRPOConfig, AutoModelForCausalLMWithValueHead
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset, Dataset
import os
from typing import Dict, List
import json
import re
import random
import types
from dataclasses import dataclass
from typing import Optional, Tuple

def extract_xml_answer(text: str) -> str:
    """Extract the final answer from the model's response."""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    """Extract the answer from GSM8K format."""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def format_prompt(question: str) -> str:
    """Format the prompt for training."""
    return f"Question: {question}\nLet's solve this step by step:\n"

def get_gsm8k_questions(system_prompt, split="train") -> Dataset:
    """Load and prepare GSM8K dataset."""
    data = load_dataset('openai/gsm8k', 'main')[split]
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': x['question']},
            {'role': 'assistant', 'content': f"<contemplating>{'.' * random.randint(4, 64)}</contemplating>"}
        ],
        'answer': extract_hash_answer(x['answer'])
    })
    return data

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Reward function based on answer correctness."""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    """Reward function for integer answers."""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function for strict format compliance."""
    pattern = r"^<contemplating>\n.*?\n</contemplating>\n<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function for soft format compliance."""
    pattern = r"<contemplating>.*?</contemplating>\s*<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def contemplating_reward_func(completions, **kwargs) -> list[float]:
    """Reward function for contemplating section format."""
    def extract_contemplating(text: str) -> str:
        if "<contemplating>" not in text or "</contemplating>" not in text:
            return ""
        hidden = text.split("<contemplating>")[-1]
        hidden = hidden.split("</contemplating>")[0]
        return hidden.strip()
    
    responses = [completion[0]["content"] for completion in completions]
    hidden_parts = [extract_contemplating(r) for r in responses]
    
    # only allow dots (periods)
    placeholder_pattern = r'^[.]*$'
    matches = [bool(re.match(placeholder_pattern, h)) for h in hidden_parts]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    """Count XML tags and penalize extra text."""
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """Reward function for XML tag counting."""
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

def train_model(
    model_name: str,
    output_dir: str,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-5,
    max_length: int = 1024,
    save_steps: int = 1000,
    logging_steps: int = 100,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    use_8bit: bool = True,
    grpo_beta: float = 0.1,
    grpo_penalty_weight: float = 0.1
):
    # Get system prompt
    system_prompt = """
    Respond in the following format:
    <contemplating>
    ...(The contemplating section should be filled with ONLY periods/dots (.) - no other characters or text allowed.)
    </contemplating>
    <reasoning>
    ...(Perform explicit reasoning in this section.)
    </reasoning>
    <answer>
    ...(Write your final answer here. Just the final number value, no text allowed.)
    </answer>
    """
    
    # Load model and tokenizer
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_8bit=use_8bit,
        use_cache=False  # Disable cache for gradient checkpointing
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    # Prepare model for LoRA training
    if use_8bit:
        model = prepare_model_for_kbit_training(model)
        # Enable gradient checkpointing with explicit use_reentrant=False
        model.gradient_checkpointing_enable(use_reentrant=False)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.warnings_issued = {}
    def add_model_tags(self, tags):
        pass
    model.add_model_tags = add_model_tags.__get__(model)
    model.generation_config = GenerationConfig.from_pretrained(model_name)
    
    # Wrap forward method to handle tuple outputs
    @dataclass
    class ModelOutput:
        logits: torch.Tensor
        value: Optional[torch.Tensor] = None
        
    original_forward = model.forward
    def wrapped_forward(self, *args, **kwargs):
        output = original_forward(*args, **kwargs)
        if isinstance(output, tuple):
            return ModelOutput(logits=output[0], value=output[1] if len(output) > 1 else None)
        return output
    model.forward = types.MethodType(wrapped_forward, model)
    
    # Prepare dataset
    dataset = get_gsm8k_questions(system_prompt, split="train")
    
    # GRPO configuration
    grpo_config = GRPOConfig(
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        beta=grpo_beta,
        seed=42,
        max_grad_norm=1.0,
        batch_size=per_device_train_batch_size,
        mini_batch_size=per_device_train_batch_size,
        optimize_cuda_cache=True,
        early_stopping=True,
        penalty_weight=grpo_penalty_weight
    )
    
    # Initialize GRPO trainer
    grpo_trainer = GRPOTrainer(
        args=grpo_config,
        model=model,
        reward_funcs=[
            correctness_reward_func,
            int_reward_func,
            strict_format_reward_func,
            soft_format_reward_func,
            contemplating_reward_func,
            xmlcount_reward_func
        ],
        processing_class=tokenizer,
        train_dataset=dataset,
    )
    
    # Training loop
    grpo_trainer.train()
    
    # Save the final model
    grpo_trainer.save_pretrained(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Name or path of the model to fine-tune")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X steps")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X steps")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--use_8bit", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--grpo_beta", type=float, default=0.1, help="GRPO beta parameter")
    parser.add_argument("--grpo_penalty_weight", type=float, default=0.1, help="GRPO penalty weight")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_model(
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_8bit=args.use_8bit,
        grpo_beta=args.grpo_beta,
        grpo_penalty_weight=args.grpo_penalty_weight
    ) 