import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from datasets import load_dataset
import json
import os
from tqdm import tqdm
import re

def extract_answer(text: str) -> str:
    """Extract the final answer from the model's response."""
    if "####" not in text:
        return ""
    return text.split("####")[1].strip()

def evaluate_model(model_path: str, output_dir: str, max_length: int = 1024):
    # Check if the model is a LoRA model
    try:
        peft_config = PeftConfig.from_pretrained(model_path)
        is_lora = True
        base_model_name = peft_config.base_model_name_or_path
    except:
        is_lora = False
        base_model_name = model_path
    
    # Load base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Load LoRA weights if it's a LoRA model
    if is_lora:
        model = PeftModel.from_pretrained(model, model_path)
        print("Loaded LoRA model from:", model_path)
    
    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    # Load GSM8K test dataset
    dataset = load_dataset("gsm8k", "main")["test"]
    
    correct = 0
    total = 0
    results = []
    
    for item in tqdm(dataset):
        # Format the prompt
        prompt = f"Question: {item['question']}\nLet's solve this step by step:\n"
        
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted_answer = extract_answer(response)
        true_answer = extract_answer(item["answer"])
        
        # Compare answers
        is_correct = predicted_answer == true_answer
        correct += int(is_correct)
        total += 1
        
        results.append({
            "question": item["question"],
            "true_answer": true_answer,
            "predicted_answer": predicted_answer,
            "full_response": response,
            "is_correct": is_correct
        })
        
        # Save intermediate results
        if total % 10 == 0:
            accuracy = (correct / total) * 100
            print(f"Progress: {total} examples, Current accuracy: {accuracy:.2f}%")
            
            with open(os.path.join(output_dir, "intermediate_results.json"), "w") as f:
                json.dump({
                    "accuracy": accuracy,
                    "results": results
                }, f, indent=2)
    
    # Save final results
    final_accuracy = (correct / total) * 100
    print(f"\nFinal Accuracy: {final_accuracy:.2f}%")
    
    with open(os.path.join(output_dir, "final_results.json"), "w") as f:
        json.dump({
            "accuracy": final_accuracy,
            "results": results
        }, f, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model to evaluate")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save evaluation results")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    evaluate_model(args.model_path, args.output_dir, args.max_length) 