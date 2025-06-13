import os
import torch
from typing import Optional, Tuple, Callable
from transformers import AutoModelForCausalLM, AutoTokenizer
from types import SimpleNamespace

from .training_utils import create_optimizer


def save_checkpoint(
    model,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_accuracy: float,
    output_dir: str,
    step: int
):
    """Save a training checkpoint containing prefix generator state and current states/prefixes."""
    # Save the prefix generator state dict
    prefix_generator_state = {
        name: param for name, param in model.state_dict().items()
        if name.startswith('prefix_generators.')
    }
    
    # Save current states and prefixes if they exist
    current_states = model.current_states.detach().cpu() if model.current_states is not None else None
    current_prefixes = model.current_prefixes.detach().cpu() if model.current_prefixes is not None else None
    
    # DEBUG: Print optimizer state info when saving
    optimizer_state = optimizer.state_dict()
    print(f"DEBUG: Saving optimizer with {len(optimizer_state['param_groups'])} parameter groups")
    if 'state' in optimizer_state:
        print(f"DEBUG: Optimizer state has {len(optimizer_state['state'])} parameter states")
    
    torch.save(
        {
            "prefix_generator_state": prefix_generator_state,
            "current_states": current_states,
            "current_prefixes": current_prefixes,
            "optimizer_state_dict": optimizer_state,
            "epoch": epoch,
            "global_step": global_step,
            "best_accuracy": best_accuracy,
            "base_model_name": model.base_model.name_or_path,  # Save the base model name
            "config": model.config  # Save the prefix generator config
        },
        os.path.join(output_dir, f"checkpoint-{step}.pt")
    )


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Find the latest checkpoint file in the output directory.
    
    Args:
        output_dir: Directory to search for checkpoints
        
    Returns:
        Path to the latest checkpoint file, or None if no checkpoints found
    """
    if not os.path.exists(output_dir):
        return None
    
    checkpoint_files = []
    for file in os.listdir(output_dir):
        if file.startswith("checkpoint-") and file.endswith(".pt"):
            # Extract step number from filename
            try:
                step = int(file.split("-")[1].split(".")[0])
                checkpoint_files.append((step, os.path.join(output_dir, file)))
            except (ValueError, IndexError):
                continue
    
    if not checkpoint_files:
        return None
    
    # Return the checkpoint with the highest step number
    checkpoint_files.sort(key=lambda x: x[0])
    return checkpoint_files[-1][1]


def find_latest_checkpoint_folder(base_output_dir: str) -> Optional[str]:
    """Find the latest checkpoint folder in the base output directory.
    
    Args:
        base_output_dir: Base directory to search for checkpoint folders
        
    Returns:
        Path to the latest checkpoint folder, or None if no checkpoint folders found
    """
    if not os.path.exists(base_output_dir):
        return None
    
    checkpoint_folders = []
    for item in os.listdir(base_output_dir):
        item_path = os.path.join(base_output_dir, item)
        if os.path.isdir(item_path):
            # Check if this folder contains any checkpoint files
            if find_latest_checkpoint(item_path) is not None:
                # Extract datetime from folder name (assuming format: model_dataset_MMDDHHMM)
                try:
                    parts = item.split('_')
                    if len(parts) >= 3:
                        datetime_str = parts[-1]  # Last part should be MMDDHHMM
                        # Convert to comparable format (simple string comparison works for MMDDHHMM)
                        checkpoint_folders.append((datetime_str, item_path))
                except (ValueError, IndexError):
                    # If we can't parse datetime, just use the folder name
                    checkpoint_folders.append((item, item_path))
    
    if not checkpoint_folders:
        return None
    
    # Return the folder with the latest datetime
    checkpoint_folders.sort(key=lambda x: x[0])
    return checkpoint_folders[-1][1]


def debug_optimizer_parameter_compatibility(model, checkpoint: dict) -> bool:
    """Debug and check compatibility between current model and saved optimizer state.
    
    Args:
        model: Current model with prefix generators
        checkpoint: Loaded checkpoint dictionary
        
    Returns:
        bool: True if optimizer states are compatible, False otherwise
    """
    print("DEBUG: Prefix generator parameters in loaded model:")
    current_prefix_params = []
    for name, param in model.named_parameters():
        if name.startswith('prefix_generators.') and param.requires_grad:
            print(f"  {name}: shape={param.shape}, numel={param.numel()}")
            current_prefix_params.append((name, param.shape, param.numel()))
    
    # Print saved optimizer state parameter info
    saved_optimizer_state = checkpoint["optimizer_state_dict"]
    print(f"DEBUG: Saved optimizer has {len(saved_optimizer_state['param_groups'])} parameter groups")
    if 'state' in saved_optimizer_state:
        print(f"DEBUG: Saved optimizer state has {len(saved_optimizer_state['state'])} parameter states")
        for param_id, param_state in saved_optimizer_state['state'].items():
            if 'exp_avg' in param_state:
                exp_avg_shape = param_state['exp_avg'].shape
                exp_avg_numel = param_state['exp_avg'].numel()
                step = param_state.get('step', 'N/A')
                print(f"  Param ID {param_id}: shape={exp_avg_shape}, numel={exp_avg_numel}, step={step}")
            else:
                print(f"  Param ID {param_id}: keys={list(param_state.keys())}")
    
    # Check if parameter counts and shapes match
    print(f"DEBUG: Current model has {len(current_prefix_params)} prefix generator parameters")
    print(f"DEBUG: Saved optimizer has {len(saved_optimizer_state.get('state', {}))} parameter states")
    
    saved_state_count = len(saved_optimizer_state.get('state', {}))
    if len(current_prefix_params) == saved_state_count:
        print("DEBUG: Parameter counts match!")
        # Compare shapes
        all_match = True
        for i, (name, shape, numel) in enumerate(current_prefix_params):
            if i in saved_optimizer_state['state']:
                saved_shape = saved_optimizer_state['state'][i]['exp_avg'].shape
                saved_numel = saved_optimizer_state['state'][i]['exp_avg'].numel()
                match = shape == saved_shape
                print(f"DEBUG: {name} - Current: {shape}({numel}) vs Saved: {saved_shape}({saved_numel}) - Match: {match}")
                if not match:
                    all_match = False
            else:
                print(f"DEBUG: {name} - No corresponding saved state found for index {i}")
                all_match = False
        return all_match
    else:
        print("DEBUG: Parameter counts do NOT match!")
        return False


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    adapter_generator_class: Callable,
):
    """Load a training checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model onto
        
    Returns:
        Tuple of (model, optimizer, epoch, global_step, best_accuracy)
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load base model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(
        checkpoint["base_model_name"],
        torch_dtype=torch.float16,  # Use float16 for memory efficiency
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint["base_model_name"])
    
    # Create model with saved config
    model = adapter_generator_class(
        base_model=base_model,
        config=checkpoint["config"],
    ).to(device)
    
    # Load prefix generator state
    model.load_state_dict(checkpoint["prefix_generator_state"], strict=False)
    
    # # no need to restore current states and prefixes since they are instance specific
    # # but if need, uncomment the following
    # if checkpoint["current_states"] is not None:
    #     model.current_states = checkpoint["current_states"].to(device)
    # if checkpoint["current_prefixes"] is not None:
    #     model.current_prefixes = checkpoint["current_prefixes"].to(device)
    #     model._set_layer_prefixes()  # Set the prefixes in the attention layers
    
    # Create optimizer and load its state - ONLY PREFIX GENERATOR PARAMETERS
    # Check optimizer parameter compatibility
    is_compatible = debug_optimizer_parameter_compatibility(model, checkpoint)
    
    # Create optimizer using the same function as new training
    # Extract learning rate from checkpoint config if available
    learning_rate = getattr(checkpoint.get("config", {}), "learning_rate", 1e-4)
    optimizer = create_optimizer(model, learning_rate)
    
    # Load optimizer state only if compatible
    if is_compatible:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("DEBUG: Successfully loaded optimizer state dict")
        except Exception as e:
            print(f"DEBUG: Failed to load optimizer state dict: {e}")
            print("DEBUG: Starting with fresh optimizer state")
    else:
        print("DEBUG: Optimizer state incompatible with current model - starting with fresh optimizer state")
        print("DEBUG: This is expected if the model architecture or parameter selection changed")
    
    # Get training state
    epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]
    best_accuracy = checkpoint["best_accuracy"]
    
    return model, tokenizer, optimizer, epoch, global_step, best_accuracy 