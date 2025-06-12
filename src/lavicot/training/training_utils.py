import torch
import torch.nn as nn
from typing import List
from transformers import PreTrainedTokenizer
from types import SimpleNamespace

from ..models.lavicot_bias import TestTimePrefixModel

def set_seed(seed: int):
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed to set
    """
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_config_value(config: SimpleNamespace, key: str, default=None):
    """Safely get a config value with a default fallback."""
    return getattr(config, key, default)


def setup_training_environment(config: SimpleNamespace) -> str:
    """Setup basic training environment including device and seed.
    
    Args:
        config: Training configuration
        
    Returns:
        device: Device string ("cuda" or "cpu")
    """
    set_seed(get_config_value(config, 'seed', 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return device


def create_optimizer(model: TestTimePrefixModel, learning_rate: float) -> torch.optim.Optimizer:
    """Create optimizer for prefix generator parameters only.
    
    Args:
        model: The model with prefix generators
        learning_rate: Learning rate for the optimizer
        
    Returns:
        Configured AdamW optimizer
    """
    # Get only prefix generator parameters
    prefix_params = list(model.prefix_generators.parameters())
    if not prefix_params:
        raise ValueError("No trainable prefix generator parameters found!")
    
    # DEBUG: Print prefix generator parameter info
    print("DEBUG: Prefix generator parameters for optimizer:")
    for name, param in model.named_parameters():
        if name.startswith('prefix_generators.') and param.requires_grad:
            print(f"  {name}: shape={param.shape}, numel={param.numel()}")
    
    print(f"DEBUG: Creating optimizer with {len(prefix_params)} parameter groups ({sum(p.numel() for p in prefix_params):,} total parameters)")
    
    return torch.optim.AdamW(prefix_params, lr=learning_rate)


def create_scheduler(
    optimizer: torch.optim.Optimizer, 
    config: SimpleNamespace,
    start_epoch: int = 0
) -> torch.optim.lr_scheduler.OneCycleLR:
    """Create learning rate scheduler.
    
    Args:
        optimizer: The optimizer to schedule
        config: Training configuration
        start_epoch: Starting epoch (for resumed training)
        
    Returns:
        Configured OneCycleLR scheduler
    """
    total_steps = (config.num_train_samples // config.per_device_train_batch_size) * config.num_train_epochs
    
    # Ensure total_steps is at least 1 to avoid division by zero
    if total_steps <= 0:
        print(f"WARNING: total_steps calculated as {total_steps}. Using 1 to avoid division by zero.")
        print(f"  num_train_samples: {config.num_train_samples}")
        print(f"  per_device_train_batch_size: {config.per_device_train_batch_size}")
        print(f"  num_train_epochs: {config.num_train_epochs}")
        total_steps = 1
    
    if start_epoch > 0:
        # Resuming: calculate remaining steps
        remaining_epochs = config.num_train_epochs - start_epoch
        remaining_steps = (config.num_train_samples // config.per_device_train_batch_size) * remaining_epochs
        
        # Ensure remaining_steps is at least 1 to avoid division by zero
        if remaining_steps <= 0:
            print(f"WARNING: remaining_steps calculated as {remaining_steps}. Using 1 to avoid division by zero.")
            remaining_steps = 1
            
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            total_steps=remaining_steps,
            pct_start=min(config.warmup_steps/remaining_steps, 1.0)
        )
    else:
        # New training: use full steps
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            pct_start=min(config.warmup_steps/total_steps, 1.0)
        )


def compute_weighted_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    question_token_lengths: List[int],
    seen_token_weight: float,
    unseen_token_weight: float
) -> torch.Tensor:
    """Compute weighted loss based on whether tokens are seen or unseen.
    
    Args:
        logits: Model output logits [batch_size, seq_len, vocab_size]
        labels: Target labels [batch_size, seq_len]
        question_token_lengths: List of question lengths for each example
        seen_token_weight: Weight for seen tokens
        unseen_token_weight: Weight for unseen tokens
        
    Returns:
        Weighted loss
    """
    batch_size, seq_len = labels.shape
    
    # Create weight mask
    weights = torch.ones_like(labels, dtype=torch.float)
    for i, q_len in enumerate(question_token_lengths):
        # Clamp question length to sequence length to avoid indexing errors
        q_len = min(q_len, seq_len)
        # Question tokens are seen
        weights[i, :q_len] = seen_token_weight
        # Answer tokens are unseen
        weights[i, q_len:] = unseen_token_weight
    
    # Create padding mask
    padding_mask = (labels != -100).float()
    
    # Compute cross entropy loss
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss = loss.view(batch_size, seq_len)
    
    # Apply weights and padding mask
    weighted_loss = loss * weights * padding_mask
    
    # Average over non-padding tokens
    return weighted_loss.sum() / padding_mask.sum() 