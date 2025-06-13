from dotmap import DotMap

from ..models.lavicot_setup import setup_base_model_and_tokenizer, setup_adapter_generator
from .training_utils import get_config_value, create_optimizer, create_scheduler
from .checkpoint_utils import load_checkpoint
from ..models.lavicot_setup import get_model_class

def setup_model_and_training_components(config: DotMap, device: str):
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
        model, tokenizer,optimizer, start_epoch, start_global_step, best_accuracy = load_checkpoint(
            resume_checkpoint_path, device
        )
        # Recreate scheduler for remaining training
        scheduler = create_scheduler(optimizer, config, start_epoch)
        
        print(f"Resumed from epoch {start_epoch}, global step {start_global_step}, best accuracy: {best_accuracy:.4f}")
    else:
        print("Starting training from scratch...")
        # Setup model and tokenizer
        print("Loading model and tokenizer...")
        base_model, tokenizer = setup_base_model_and_tokenizer(config.model_name, device)
        
        # Get the appropriate model class based on config
        model_class = get_model_class(config.model_type)
        print(f"Using model implementation: {config.model_type}")
        
        # Create model instance with the selected implementation
        model = setup_adapter_generator(base_model, model_class, device, config, tokenizer)
        
        # Create optimizer and scheduler
        optimizer = create_optimizer(model, config.learning_rate)
        scheduler = create_scheduler(optimizer, config)
    
    return model, optimizer, scheduler, tokenizer, start_epoch, start_global_step, best_accuracy

