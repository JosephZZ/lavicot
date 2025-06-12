import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import PreTrainedTokenizer, PreTrainedModel
import random
from typing import Callable
import re
import os
import datetime
import wandb
from types import SimpleNamespace

from ..models.lavicot_bias import TestTimePrefixModel


def get_partial_sequence(sequence: str, max_length: int, proportion: float = 0.5) -> Tuple[str, int]:
    """Get a partial sequence based on proportion."""
    target_length = int(len(sequence) * proportion)
    target_length = min(target_length, max_length)
    partial_seq = sequence[:target_length]
    return partial_seq, target_length

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count trainable and non-trainable parameters in the model."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = trainable_params + non_trainable_params
    
    return {
        "trainable": trainable_params,
        "non_trainable": non_trainable_params,
        "total": total_params
    }

def get_gpu_memory_usage() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0

def get_gradient_norm(model: nn.Module, parameters=None) -> float:
    """Calculate the total gradient norm across specified parameters.
    
    Args:
        model: The model to calculate gradient norm for
        parameters: Optional iterable of parameters to check. If None, uses all model parameters.
    
    Returns:
        Total gradient norm
    """
    total_norm = 0.0
    param_iter = parameters if parameters is not None else model.parameters()
    for p in param_iter:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def get_parameter_norm(model: nn.Module) -> float:
    """Calculate the total parameter norm."""
    total_norm = 0.0
    for p in model.parameters():
        if p.requires_grad:
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def get_token_statistics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """Calculate token-level statistics."""
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == labels).float().mean().item()
    
    # Calculate perplexity
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    perplexity = torch.exp(loss).item()
    
    # Calculate token distribution statistics
    token_probs = F.softmax(logits, dim=-1)
    entropy = -(token_probs * torch.log(token_probs + 1e-10)).sum(dim=-1).mean().item()
    
    return {
        "token_accuracy": correct,
        "perplexity": perplexity,
        "token_entropy": entropy
    }

def get_config_value(config: SimpleNamespace, key: str, default=None):
    """Safely get a config value with a default fallback."""
    return getattr(config, key, default)


def initialize_wandb_tracking(config: SimpleNamespace, model: TestTimePrefixModel) -> None:
    """Initialize WandB tracking if enabled.
    
    Args:
        config: Training configuration
        model: The model being trained (for generating run names)
    """
    is_use_wandb = get_config_value(config, 'use_wandb', False)
    if not is_use_wandb:
        return
    
    # Check if we're resuming from a checkpoint
    resume_checkpoint_path = get_config_value(config, 'resume_checkpoint_path', None)
    is_resuming = resume_checkpoint_path is not None
    
    # Get base model and dataset names for naming
    base_model_name = config.model_name.split('/')[-1]
    dataset_name = config.dataset_name
    
    if is_resuming:
        # Try to resume existing WandB run
        dir_name = os.path.basename(config.output_dir)
        
        # Try to resume wandb run by ID if we can find it
        wandb_resume_id = None
        if os.path.exists(config.output_dir):
            for file in os.listdir(config.output_dir):
                if file.startswith('wandb-resume-') and file.endswith('.txt'):
                    with open(os.path.join(config.output_dir, file), 'r') as f:
                        wandb_resume_id = f.read().strip()
                    break
        
        # Generate run name from directory if not already set
        if not hasattr(config, 'wandb_run_name') or not config.wandb_run_name:
            config.wandb_run_name = dir_name
        
        try:
            if wandb_resume_id:
                wandb.init(
                    project=get_config_value(config, 'wandb_project', 'lavicot'),
                    entity=get_config_value(config, 'wandb_entity', None),
                    id=wandb_resume_id,
                    resume="must",
                    config=vars(config)
                )
                print(f"Resumed wandb run with ID: {wandb_resume_id}")
            else:
                # Start new wandb run but with existing name
                wandb.init(
                    project=get_config_value(config, 'wandb_project', 'lavicot'),
                    entity=get_config_value(config, 'wandb_entity', None),
                    name=config.wandb_run_name,
                    config=vars(config)
                )
                print(f"Started new wandb run with existing name: {config.wandb_run_name}")
        except Exception as e:
            print(f"Failed to resume wandb run: {e}")
            # Fall back to new run
            wandb.init(
                project=get_config_value(config, 'wandb_project', 'lavicot'),
                entity=get_config_value(config, 'wandb_entity', None),
                name=config.wandb_run_name,
                config=vars(config)
            )
    else:
        # Initialize new wandb run
        # Generate descriptive run name if not provided
        if not hasattr(config, 'wandb_run_name') or not config.wandb_run_name:
            # Get datetime string for new runs
            current_time = datetime.datetime.now()
            datetime_str = current_time.strftime("%m%d%H%M")
            
            # Shared weights info
            shared_weights = "sw" if config.prefix_generator['shared_weight_for_all_layers'] else "nosw"
            
            # Layer selection info
            layer_mode = config.prefix_generator['layer_selection_mode']
            if layer_mode == "specific":
                layer_selection = config.prefix_generator['layer_selection']
                if isinstance(layer_selection, list):
                    layer_str = f"layers_{min(layer_selection)}-{max(layer_selection)}" if len(layer_selection) > 1 else f"layer_{layer_selection[0]}"
                else:
                    layer_str = f"layer_{layer_selection}"
            elif layer_mode == "all":
                layer_str = "layers_all"
            else:
                layer_str = f"layers_{layer_mode}"
            
            config.wandb_run_name = f"{base_model_name}_{dataset_name}_lr{config.learning_rate}_r{config.num_rounds}_iter{config.prefix_generator['max_iterations']}_{shared_weights}_{layer_str}_{datetime_str}"
        
        wandb.init(
            project=get_config_value(config, 'wandb_project', 'lavicot'),
            entity=get_config_value(config, 'wandb_entity', None),
            name=config.wandb_run_name,
            config=vars(config)
        )
        
        # Save wandb run ID for future resumption
        if wandb.run and wandb.run.id:
            with open(os.path.join(config.output_dir, f'wandb-resume-{wandb.run.id}.txt'), 'w') as f:
                f.write(wandb.run.id)


def print_training_configuration(config: SimpleNamespace, train_indices: List[int], eval_instances: List[Dict], full_eval_instances: List[Dict], start_global_step: int) -> None:
    """Print comprehensive training configuration information.
    
    Args:
        config: Training configuration
        train_indices: List of training sample indices
        eval_instances: List of evaluation instances for during-training eval
        full_eval_instances: List of all test instances for final evaluation
        start_global_step: Starting global step (for resumed training)
    """
    # Calculate total steps
    total_steps = (config.num_train_samples // config.per_device_train_batch_size) * config.num_train_epochs
    print(f"Training on {len(train_indices)} samples from {config.dataset_name}")
    print(f"Training configuration:")
    print(f"  Batch size: {config.per_device_train_batch_size}")
    print(f"  Gradient accumulation steps: {config.gradient_accumulation_steps}")
    print(f"  Effective size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print(f"  Number of epochs: {config.num_train_epochs}")
    print(f"  Total steps to train: {total_steps}")
    print(f"  Start global step: {start_global_step}")
    print(f"Evaluating during training on {config.eval_during_training_samples} samples")
    print(f"Final evaluation will use full test set ({len(full_eval_instances)} samples)")

    print(f"\nUsing {'stochastic' if config.stochastic_rounds else 'consistent'} rounds (with max {config.num_rounds})")
    print(f"First round question only: {config.first_round_question_only}")
    print(f"Use previous prefix: {config.use_previous_prefix}")
    print(f"Token weights - Seen: {config.seen_token_weight}, Unseen: {config.unseen_token_weight}")
    print(f"Proportion range: {config.proportion_min:.1f} to {config.proportion_max:.1f}")
    print(f"Max prefix generation iterations: {config.prefix_generator['max_iterations']}")
    print(f"Last N iterations that has gradients: {config.prefix_generator['gradient_steps']}")

