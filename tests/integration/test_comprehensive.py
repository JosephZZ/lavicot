#!/usr/bin/env python3
"""
Comprehensive LaViCoT System Test
Tests all critical aspects: base model, parameter freezing, and actual training.
"""

import os
import sys
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Path is handled by pytest configuration

from src.lavicot.models.base_model_integration import setup_model_and_tokenizer, setup_prefix_generator
from src.lavicot.utils.data_utils import prepare_batch
from src.lavicot.config.config_loader import load_config

def test_base_model():
    """Test 1: Base model generates meaningful text without modifications."""
    print("=" * 80)
    print("TEST 1: BASE MODEL VERIFICATION")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "distilgpt2"
    
    print(f"Device: {device}")
    print(f"Loading {model_name} (base model only)...")
    
    # Load base model without any modifications
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test generation quality
    test_prompts = [
        "Question: What is 2 + 2?\nAnswer:",
        "Question: If I have 5 apples and eat 2, how many do I have left?\nLet me think:"
    ]
    
    generation_quality = 0
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}: {prompt}")
        
        input_ids = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids["input_ids"],
                attention_mask=input_ids["attention_mask"],
                max_length=input_ids["input_ids"].shape[1] + 30,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=2
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        continuation = generated_text[len(prompt):].strip()
        print(f"Generated: {continuation}")
        
        # Quality check
        if len(continuation) > 10 and not all(c in ' -$(),' for c in continuation):
            print("✓ Meaningful generation")
            generation_quality += 1
        else:
            print("⚠ Low quality generation")
    
    success = generation_quality == len(test_prompts)
    print(f"\nBase Model Test: {'✅ PASSED' if success else '❌ FAILED'}")
    return success


def test_parameter_freezing():
    """Test 2: Only prefix generators are trainable, base model is frozen."""
    print("\n" + "=" * 80)
    print("TEST 2: PARAMETER FREEZING VERIFICATION")
    print("=" * 80)
    
    config = load_config("experiments/configs/improved_config.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading model with prefix generators...")
    
    # Load model with prefix generators (should freeze base model)
    base_model, tokenizer = setup_model_and_tokenizer(config.model_name, device)
    model = setup_prefix_generator(base_model, device, vars(config), tokenizer)
    
    # Analyze parameters
    total_params = 0
    trainable_params = 0
    base_model_trainable = 0
    base_model_total = 0
    prefix_generator_trainable = 0
    prefix_generator_total = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        
        if param.requires_grad:
            trainable_params += param.numel()
        
        if name.startswith('base_model.'):
            base_model_total += param.numel()
            if param.requires_grad:
                base_model_trainable += param.numel()
        elif name.startswith('prefix_generators.'):
            prefix_generator_total += param.numel()
            if param.requires_grad:
                prefix_generator_trainable += param.numel()
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({(trainable_params/total_params)*100:.1f}%)")
    print(f"Base model: {base_model_total:,} total, {base_model_trainable:,} trainable")
    print(f"Prefix generators: {prefix_generator_total:,} total, {prefix_generator_trainable:,} trainable")
    
    # Test generation with prefixes
    print("\nTesting generation with prefix generators...")
    test_prompt = "Question: What is 3 + 4?\nAnswer:"
    
    model.reset_prefixes()
    input_ids = tokenizer(test_prompt, return_tensors="pt").to(device)
    model.update_prefix_given_input(input_ids["input_ids"])
    
    print(f"Prefix shape: {model.current_prefixes.shape if model.current_prefixes is not None else 'None'}")
    
    # Validation
    base_frozen = (base_model_trainable == 0)
    prefix_trainable = (prefix_generator_trainable > 0)
    only_prefix_trainable = (trainable_params == prefix_generator_trainable)
    
    print(f"\nBase model frozen: {'✓' if base_frozen else '✗'}")
    print(f"Prefix generators trainable: {'✓' if prefix_trainable else '✗'}")
    print(f"Only prefix generators trainable: {'✓' if only_prefix_trainable else '✗'}")
    
    success = base_frozen and prefix_trainable and only_prefix_trainable
    print(f"\nParameter Freezing Test: {'✅ PASSED' if success else '❌ FAILED'}")
    return success, model, tokenizer


def test_actual_training(model, tokenizer):
    """Test 3: Training actually updates prefix generator parameters."""
    print("\n" + "=" * 80)
    print("TEST 3: ACTUAL TRAINING VERIFICATION")
    print("=" * 80)
    
    device = next(model.parameters()).device
    
    # Get initial parameter values
    initial_params = {}
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            initial_params[name] = param.data.clone()
            trainable_params.append((name, param))
    
    print(f"Tracking {len(initial_params)} trainable parameters")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW([p for _, p in trainable_params], lr=1e-4)
    
    # Load training data
    dataset = load_dataset("gsm8k", "main")
    train_data = [dataset["train"][i] for i in range(3)]  # Just 3 samples for quick test
    
    # Training loop
    model.train()
    losses = []
    
    for step in range(2):  # Just 2 training steps
        print(f"\nTraining Step {step + 1}:")
        
        optimizer.zero_grad()
        model.reset_prefixes()
        
        # Round 1: Question only
        batch_inputs_q = prepare_batch(
            prep_mode="question_only",
            data_instances=train_data,
            tokenizer=tokenizer,
            max_length=256,
            device=device
        )
        model.update_prefix_given_input(input_ids=batch_inputs_q)
        
        # Round 2: Partial reasoning  
        batch_inputs_partial = prepare_batch(
            prep_mode="cot_only",
            data_instances=train_data,
            tokenizer=tokenizer,
            max_length=256,
            min_proportion=0.3,
            max_proportion=0.7,
            device=device
        )
        model.update_prefix_given_input(input_ids=batch_inputs_partial)
        
        # Full sequence for loss
        full_sequences = prepare_batch(
            prep_mode="full",
            data_instances=train_data,
            tokenizer=tokenizer,
            max_length=256,
            device=device
        )
        
        # Forward pass
        outputs = model(
            input_ids=full_sequences,
            attention_mask=torch.ones_like(full_sequences),
            labels=full_sequences
        )
        
        # Compute loss
        loss = nn.CrossEntropyLoss()(
            outputs.logits.view(-1, outputs.logits.size(-1)),
            full_sequences.view(-1)
        )
        
        print(f"  Loss: {loss.item():.4f}")
        losses.append(loss.item())
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        grad_count = sum(1 for _, p in trainable_params if p.grad is not None)
        total_grad_norm = sum(p.grad.norm().item() for _, p in trainable_params if p.grad is not None)
        
        print(f"  Gradients: {grad_count}/{len(trainable_params)} parameters")
        print(f"  Gradient norm: {total_grad_norm:.3f}")
        
        optimizer.step()
    
    # Check parameter changes
    changed_params = 0
    total_change = 0
    
    for name, param in trainable_params:
        if name in initial_params:
            change = (param.data - initial_params[name]).abs().sum().item()
            total_change += change
            if change > 1e-8:
                changed_params += 1
    
    print(f"\nParameter Updates:")
    print(f"  Changed parameters: {changed_params}/{len(trainable_params)}")
    print(f"  Total change magnitude: {total_change:.6f}")
    print(f"  Loss trend: {losses[0]:.4f} → {losses[-1]:.4f}")
    
    # Validation
    loss_decreased = losses[-1] < losses[0]
    params_updated = changed_params > 0
    significant_change = total_change > 1e-6
    
    print(f"\nLoss decreased: {'✓' if loss_decreased else '✗'}")
    print(f"Parameters updated: {'✓' if params_updated else '✗'}")
    print(f"Significant changes: {'✓' if significant_change else '✗'}")
    
    success = loss_decreased and params_updated and significant_change
    print(f"\nActual Training Test: {'✅ PASSED' if success else '❌ FAILED'}")
    return success


def main():
    """Run comprehensive LaViCoT system tests."""
    print("🧪 COMPREHENSIVE LaViCoT SYSTEM TEST")
    print("Testing: Base Model → Parameter Freezing → Actual Training")
    
    # Test 1: Base model verification
    test1_passed = test_base_model()
    
    # Test 2: Parameter freezing verification  
    test2_passed, model, tokenizer = test_parameter_freezing()
    
    # Test 3: Actual training verification
    test3_passed = test_actual_training(model, tokenizer) if test2_passed else False
    
    # Final results
    print("\n" + "=" * 80)
    print("COMPREHENSIVE TEST RESULTS")
    print("=" * 80)
    
    print(f"1. Base Model Generation:     {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"2. Parameter Freezing:        {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print(f"3. Actual Training:           {'✅ PASSED' if test3_passed else '❌ FAILED'}")
    
    all_passed = test1_passed and test2_passed and test3_passed
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! LaViCoT system is ready for training.")
        print("You can now run: python train_lavicot/train.py --config config/improved_config.yaml")
    else:
        print("\n❌ SOME TESTS FAILED! Please check the issues above.")
    
    return all_passed


if __name__ == "__main__":
    main() 