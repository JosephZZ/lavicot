import torch
from typing import Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from ..utils.tokenizer_utils import setup_padding_token
from ..utils.logging_utils import count_parameters

from ..models.lavicot_models.lavicot_bias import TestTimeGammaBiasModel
from ..models.lavicot_models.lavicot_bias_gamma import TestTimeGammaBiasModel
from ..models.lavicot_models.lavicot_prefix_attention import TestTimePrefixAttentionGammaModel

def get_model_class(model_type: str):
    """Get the appropriate model class based on the model type.
    
    Args:
        model_type: String specifying which model implementation to use
        
    Returns:
        The appropriate TestTimePrefixModel class
    """
    model_classes = {
        "lavicot_bias": TestTimeGammaBiasModel,
        "lavicot_bias_gamma": TestTimeGammaBiasModel,
        "lavicot_prefix_attention": TestTimePrefixAttentionGammaModel,
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}. Available types: {list(model_classes.keys())}")
        
    return model_classes[model_type]

def setup_base_model_and_tokenizer(
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


def setup_adapter_generator(
    base_model: PreTrainedModel,
    generator_model_class,
    device: str,
    config_dict,
    initial_prefixes: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    freeze_base_model: bool = True
):
    """Initialize and configure the prefix generator.
    
    Args:
        base_model: The base model to wrap
        device: Device to place the model on
        config_dict: Configuration dictionary containing prefix generator settings
        initial_prefixes: Optional initial prefixes tensor [num_layers, ...]
        initial_states: Optional initial states tensor [num_layers, ...]
    """
    # Get prefix generator config from config_dict
    prefix_config = config_dict.get('prefix_generator', {})

    model = generator_model_class(base_model, prefix_config) 
    
    # Set initial prefixes and states if provided
    if initial_prefixes is not None:
        model.current_prefixes = initial_prefixes
    if initial_states is not None:
        model.current_states = initial_states
    
    model.to(device)
    

    if freeze_base_model:
        # FREEZE BASE MODEL PARAMETERS - Only prefix generators should be trainable
        print("Freezing base model parameters...")
        frozen_count = 0
        for name, param in base_model.named_parameters():
            param.requires_grad = False
            frozen_count += param.numel()
        print(f" {frozen_count:,} base model parameters")

    # Count final parameters after freezing and adding prefix generators
    final_param_counts = count_parameters(model)
    

    print(f"\nFinal Model Configuration:")
    print(f"  Total parameters: {final_param_counts['total']:,}")
    print(f"  Trainable parameters: {final_param_counts['trainable']:,} ({(final_param_counts['trainable'] / final_param_counts['total'] * 100):.1f}%)")
    print(f"  Frozen parameters: {final_param_counts['non_trainable']:,}")
    
    return model 