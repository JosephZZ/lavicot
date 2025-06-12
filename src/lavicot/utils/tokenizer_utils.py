"""
Utilities for handling tokenizer padding across different model architectures.
"""
from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import Tuple

def setup_padding_token(tokenizer: PreTrainedTokenizer, model: PreTrainedModel) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """
    Properly setup padding token for different model architectures.
    
    Args:
        tokenizer: The tokenizer to configure
        model: The model to configure
        
    Returns:
        Tuple of (configured_tokenizer, configured_model)
    """
    # Check if tokenizer already has a proper pad token
    if tokenizer.pad_token is not None and tokenizer.pad_token_id is not None:
        # Already configured, ensure model config matches
        model.config.pad_token_id = tokenizer.pad_token_id
        return tokenizer, model
    
    # Model-specific pad token handling
    model_name = model.config.model_type.lower()
    
    if model_name in ['gpt2', 'gpt']:
        # GPT-2 style models: use EOS as pad token
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
        
    elif model_name in ['llama', 'llama2']:
        # LLaMA models: try to use existing pad token, fallback to unk
        if hasattr(tokenizer, 'unk_token') and tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
            model.config.pad_token_id = tokenizer.unk_token_id
        else:
            # Fallback to EOS token
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id
            
    elif model_name in ['qwen', 'qwen2']:
        # Qwen models: usually have pad token defined, but check
        if tokenizer.pad_token is None:
            # Try to find a pad-like token first
            if '<|PAD_TOKEN|>' in tokenizer.get_vocab():
                tokenizer.pad_token = '<|PAD_TOKEN|>'
            elif '<pad>' in tokenizer.get_vocab():
                tokenizer.pad_token = '<pad>'
            else:
                # Fallback to EOS token
                tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        
    else:
        # Generic fallback: use EOS token as pad token
        print(f"Warning: Unknown model type '{model_name}', using EOS token as pad token")
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    # Verify setup
    assert tokenizer.pad_token is not None, "Failed to set pad token"
    assert tokenizer.pad_token_id is not None, "Failed to set pad token ID"
    assert model.config.pad_token_id == tokenizer.pad_token_id, "Model and tokenizer pad token ID mismatch"
    
    print(f"Configured padding token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id}) for {model_name}")
    
    return tokenizer, model

def get_generation_pad_token_id(tokenizer: PreTrainedTokenizer) -> int:
    """
    Get the appropriate pad token ID for generation.
    
    Args:
        tokenizer: The tokenizer
        
    Returns:
        Pad token ID to use for generation
    """
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    else:
        # Fallback to EOS token
        return tokenizer.eos_token_id 