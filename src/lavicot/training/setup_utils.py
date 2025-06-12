import random
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, PreTrainedTokenizer
from datasets import load_dataset
from types import SimpleNamespace

from ..models.lavicot_bias import TestTimePrefixModel
from ..models.model_setup import setup_model_and_tokenizer, setup_prefix_generator
from .training_utils import get_config_value, create_optimizer, create_scheduler
from .checkpoint_utils import load_checkpoint


def setup_model_and_training_components(config: SimpleNamespace, device: str) -> Tuple[TestTimePrefixModel, Any, Any, PreTrainedTokenizer, int, int, float]:
    """Setup model, optimizer, scheduler and tokenizer based on config.
    
    Args:
        config: Training configuration
        device: Device to load model on
        
    Returns:
        Tuple of (model, optimizer, scheduler, tokenizer, start_epoch, start_global_step, best_accuracy)
    """
    # Check if we're resuming from a checkpoint
    resume_checkpoint_path = get_config_value(config, 'resume_checkpoint_path', None)
    is_resuming = resume_checkpoint_path is not None
    
    # Initialize training state
    start_epoch = 0
    start_global_step = 0
    best_accuracy = 0.0
    
    if is_resuming:
        print(f"Loading model from checkpoint: {resume_checkpoint_path}")
        model, optimizer, start_epoch, start_global_step, best_accuracy = load_checkpoint(
            resume_checkpoint_path, device
        )
        tokenizer = AutoTokenizer.from_pretrained(model.base_model.name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.base_model.config.pad_token_id = tokenizer.eos_token_id
        
        # Recreate scheduler for remaining training
        scheduler = create_scheduler(optimizer, config, start_epoch)
        
        print(f"Resumed from epoch {start_epoch}, global step {start_global_step}, best accuracy: {best_accuracy:.4f}")
    else:
        print("Starting training from scratch...")
        # Setup model and tokenizer
        print("Loading model and tokenizer...")
        base_model, tokenizer = setup_model_and_tokenizer(config.model_name, device)
        model = setup_prefix_generator(base_model, device, vars(config), tokenizer)
        
        # Create optimizer and scheduler
        optimizer = create_optimizer(model, config.learning_rate)
        scheduler = create_scheduler(optimizer, config)
    
    return model, optimizer, scheduler, tokenizer, start_epoch, start_global_step, best_accuracy


def prepare_datasets_and_samples(config: SimpleNamespace) -> Tuple[List[int], List[Dict], List[Dict], Any, Any]:
    """Load datasets and prepare training/evaluation samples.
    
    Args:
        config: Training configuration
        
    Returns:
        Tuple of (train_indices, eval_instances, full_eval_instances, train_dataset, test_dataset)
    """
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(config.dataset_name, config.dataset_config)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    
    # Sample training data indices (will convert to instances during batching)
    train_indices = random.sample(range(len(train_dataset)), config.num_train_samples)
    
    # Prepare evaluation data - keep as instances
    eval_instances = []
    if config.eval_during_training_fixed:
        eval_indices = random.sample(range(len(test_dataset)), config.eval_during_training_samples)
        eval_instances = [test_dataset[i] for i in eval_indices]
    
    # Full test set for final evaluation
    full_eval_instances = [test_dataset[i] for i in range(len(test_dataset))]
    
    return train_indices, eval_instances, full_eval_instances, train_dataset, test_dataset 