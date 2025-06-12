import torch
from typing import Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from .lavicot_bias import (
    add_instance_level_prefix_generator,
    create_test_time_prefix_config,
    TestTimePrefixModel
)
from ..utils.tokenizer_utils import setup_padding_token
from ..utils.logging_utils import count_parameters


def setup_model_and_tokenizer(
    model_name: str,
    device: str
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Initialize model and tokenizer with proper configuration."""
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Setup tokenizer padding (robust across model architectures)
    tokenizer, base_model = setup_padding_token(tokenizer, base_model)
    
    # Count parameters (before any modifications)
    param_counts = count_parameters(base_model)
    print(f"\nBase Model Loaded: {param_counts['total']:,} parameters")
    
    return base_model, tokenizer


def setup_prefix_generator(
    base_model: PreTrainedModel,
    device: str,
    config_dict: dict,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    initial_prefixes: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None
) -> TestTimePrefixModel:
    """Initialize and configure the prefix generator.
    
    Args:
        base_model: The base model to wrap
        device: Device to place the model on
        config_dict: Configuration dictionary containing prefix generator settings
        tokenizer: Optional tokenizer for the model
        initial_prefixes: Optional initial prefixes tensor [num_layers, ...]
        initial_states: Optional initial states tensor [num_layers, ...]
    """
    # FREEZE BASE MODEL PARAMETERS - Only prefix generators should be trainable
    print("Freezing base model parameters...")
    frozen_count = 0
    for name, param in base_model.named_parameters():
        param.requires_grad = False
        frozen_count += param.numel()
    print(f"Froze {frozen_count:,} base model parameters")
    
    # Get prefix generator config from config_dict
    prefix_config = config_dict.get('prefix_generator', {})
    
    config = create_test_time_prefix_config(
        layer_selection_mode=prefix_config.get('layer_selection_mode', 'all'),
        layer_selection=prefix_config.get('layer_selection', None),
        hidden_size=prefix_config.get('rnn_hidden_size', None),
        max_iterations=prefix_config.get('max_iterations', 10),
        gradient_steps=prefix_config.get('gradient_steps', 4),
        shared_weight_for_all_layers=prefix_config.get('shared_weight_for_all_layers', False),
        use_hooks_during_prefix_update=prefix_config.get('use_hooks_during_prefix_update', False)
    )
    
    model = add_instance_level_prefix_generator(
        base_model, 
        config, 
        tokenizer
    )
    
    # Set initial prefixes and states if provided
    if initial_prefixes is not None:
        model.current_prefixes = initial_prefixes
    if initial_states is not None:
        model.current_states = initial_states
    
    model.to(device)
    
    # Count final parameters after freezing and adding prefix generators
    final_param_counts = count_parameters(model)
    
    # Count prefix generator parameters specifically
    prefix_generator_params = sum(p.numel() for p in model.prefix_generators.parameters())
    
    print(f"\nFinal Model Configuration:")
    print(f"  Total parameters: {final_param_counts['total']:,}")
    print(f"  Trainable parameters: {final_param_counts['trainable']:,} ({(final_param_counts['trainable'] / final_param_counts['total'] * 100):.1f}%)")
    print(f"  Frozen parameters: {final_param_counts['non_trainable']:,}")
    print(f"  Prefix generator parameters: {prefix_generator_params:,}")
    
    # Verify freezing worked
    trainable_base_params = sum(p.numel() for name, p in model.named_parameters() 
                               if name.startswith('base_model.') and p.requires_grad)
    if trainable_base_params > 0:
        print(f"⚠ WARNING: {trainable_base_params:,} base model parameters are still trainable!")
    else:
        print("✓ Base model successfully frozen")
    
    return model 