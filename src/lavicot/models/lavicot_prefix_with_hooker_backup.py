import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from transformers import PreTrainedModel, PreTrainedTokenizer
import numpy as np

@dataclass
class TestTimePrefixConfig:
    """Configuration for test-time prefix generation."""
    # Layer selection configuration
    layer_selection_mode: str = "all"  # "all", "specific", "exclude", or "spread"
    layer_selection: Optional[Union[List[int], int]] = None  # List of layers or number of layers to spread
    
    # Prefix generation configuration
    hidden_size: Optional[int] = None  # If None, uses model's hidden size
    max_iterations: int = 10
    gradient_steps: int = 4  # Number of iterations to allow gradients through
    shared_weight_for_all_layers: bool = True  # Whether all layers share the same prefix generator
    
    # Hook control configuration
    use_hooks_during_prefix_update: bool = False  # Whether to use hooks during prefix generation

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate RMS
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # Normalize and scale
        return x * rms * self.weight

class PrefixGenerator(nn.Module):
    """Prefix generator using iterative attention and MLP."""
    def __init__(self, config: TestTimePrefixConfig, model_hidden_size: int):
        super().__init__()
        self.hidden_size = config.hidden_size or model_hidden_size
        self.model_hidden_size = model_hidden_size
        self.gradient_steps = config.gradient_steps
        
        # Cross attention
        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        
        # MLP for transforming cross-attended context
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, 4 * self.hidden_size),
            nn.GELU(),
            nn.Linear(4 * self.hidden_size, self.hidden_size)
        )
        
        # Output projection back to model's hidden size
        self.state_projection = nn.Linear(self.hidden_size, model_hidden_size)
        
        # Layer normalizations
        self.norm1 = RMSNorm(self.hidden_size)  # Pre-norm for cross-attention
        self.norm2 = RMSNorm(self.hidden_size)  # Pre-norm for MLP
        self.norm3 = RMSNorm(self.hidden_size)  # Final norm before projection
        self.output_norm = RMSNorm(model_hidden_size)  # Final norm after projection

    def state_to_prefix_projection(self, s: torch.Tensor) -> torch.Tensor:
        """Project the state to the model's hidden size."""
        return self.output_norm(self.state_projection(self.norm3(s)))
        
    def forward(
        self,
        num_iterations: int = None,
        context: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to generate prefixes.
        
        Args:
            num_iterations: Number of iterations to perform
            context: Context sequence tensor [batch_size, seq_len, hidden_size]
            initial_state: Optional initial state tensor [batch_size, hidden_size]
            
        Returns:
            Tuple of (outputs, final_state)
            - outputs: [batch_size, num_iterations, hidden_size]
            - final_state: [batch_size, hidden_size]
        """
        batch_size = 1  # Single batch for prefix generation
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        
        # Initialize state from provided initial state or scaled Gaussian
        if initial_state is not None:
            s = initial_state.to(device=device, dtype=dtype)
        else:
            # Use principled initialization scaled by sqrt(hidden_size)
            init_std = 1.0 / np.sqrt(self.hidden_size)
            s = torch.randn(
                batch_size,
                self.hidden_size,
                device=device,
                dtype=dtype
            ) * init_std
        
        # Process through iterations
        states_iters = []
        for i in range(num_iterations):
            # Pre-norm for cross-attention
            s_norm = self.norm1(s)
            
            # Cross attention - context is now [batch_size, seq_len, hidden_size]
            q = self.query(s_norm).unsqueeze(1)  # [batch_size, 1, hidden_size]
            k = self.key(context)                 # [batch_size, seq_len, hidden_size]
            v = self.value(context)               # [batch_size, seq_len, hidden_size]
            
            # Attention scores: q @ k^T
            scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.hidden_size)  # [batch_size, 1, seq_len]
            attn_weights = F.softmax(scores, dim=-1)
            
            # Apply attention to get attended context
            context_attended = torch.matmul(attn_weights, v).squeeze(1)  # [batch_size, hidden_size]
            
            # Residual connection after cross-attention
            s = s + context_attended
            
            # Pre-norm for MLP
            s_norm = self.norm2(s)
            
            # MLP and residual connection
            s = s + self.mlp(s_norm)
            
            # Store state for this iteration
            states_iters.append(s)
            
            # Only keep gradients for the last gradient_steps iterations
            if i < num_iterations - self.gradient_steps:
                s = s.detach()
        
        # Stack outputs
        states_iters = torch.stack(states_iters, dim=1)  # [batch_size, num_iterations, hidden_size]
        
        # Project to model's hidden size and final norm
        prefix = self.state_to_prefix_projection(s)
        
        return states_iters, prefix

class PrefixAttentionHook:
    """Handles prefix injection and removal for a specific attention layer."""
    
    def __init__(self, layer_idx: int, hidden_size: int):
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.prefix = None  # Store prefix locally for proper with/without control
        self.hook_handle = None
        
    def set_prefix(self, prefix: torch.Tensor):
        """Set the prefix for this layer."""
        self.prefix = prefix
        
    def clear_prefix(self):
        """Clear the prefix for this layer.""" 
        self.prefix = None
    
    def _modify_attention_inputs(self, module, input_args):
        """Hook function to modify attention inputs by adding prefix."""
        # DEBUG: Check what we're actually receiving
        print(f"DEBUG Hook Layer {self.layer_idx}: Called with module={type(module)}")
        print(f"  input_args type={type(input_args)}, length={len(input_args) if hasattr(input_args, '__len__') else 'no len'}")
        
        # For PyTorch pre_hooks, input_args should be a tuple of arguments
        # But sometimes it might be passed differently
        if hasattr(input_args, '__len__') and len(input_args) > 0:
            for i, arg in enumerate(input_args):
                if hasattr(arg, 'shape'):
                    print(f"  arg[{i}]: {type(arg)} shape={arg.shape}")
                else:
                    print(f"  arg[{i}]: {type(arg)}")
        else:
            print(f"  input_args is empty or not iterable")
        
        # Let's also check if input is passed as individual arguments
        print(f"  Total args passed to hook: {len(input_args) if hasattr(input_args, '__len__') else 0}")
        
        if self.prefix is None:
            print(f"DEBUG Hook Layer {self.layer_idx}: No prefix set, returning original args")
            return input_args
        
        # Extract arguments based on length
        if len(input_args) == 0:
            print(f"DEBUG Hook Layer {self.layer_idx}: No input arguments received!")
            return input_args
            
        hidden_states = input_args[0]
        if hidden_states is None:
            return input_args
        
        try:
            # Convert prefix to match hidden states
            prefix_converted = self.prefix.to(
                dtype=hidden_states.dtype, 
                device=hidden_states.device
            )
            
            # Concatenate prefix with hidden states
            modified_hidden_states = torch.cat([prefix_converted, hidden_states], dim=1)
            
            # Handle different attention layer signatures
            if len(input_args) == 1:
                # Simple case: just hidden states
                return (modified_hidden_states,)
            
            elif len(input_args) >= 3:
                # Qwen-style: (hidden_states, position_embeddings, attention_mask, ...)
                position_embeddings = input_args[1]
                attention_mask = input_args[2]
                
                # Modify attention mask to account for prefix tokens
                modified_attention_mask = attention_mask
                if attention_mask is not None and attention_mask.dim() >= 2:
                    batch_size = hidden_states.shape[0]
                    prefix_len = prefix_converted.shape[1]
                    
                    # Create mask for prefix tokens (all ones - prefix tokens can attend to everything)
                    prefix_mask = torch.ones(
                        batch_size, prefix_len, 
                        dtype=attention_mask.dtype, 
                        device=attention_mask.device
                    )
                    
                    # Concatenate prefix mask with original mask
                    modified_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
                
                # Keep position embeddings unchanged (they should work with longer sequences)
                # Reconstruct args tuple with modifications
                modified_args = [modified_hidden_states, position_embeddings, modified_attention_mask]
                
                # Add remaining arguments unchanged
                if len(input_args) > 3:
                    modified_args.extend(input_args[3:])
                
                return tuple(modified_args)
            
            else:
                # Two arguments - handle as best we can
                modified_args = [modified_hidden_states]
                modified_args.extend(input_args[1:])
                return tuple(modified_args)
            
        except Exception as e:
            # Fail silently to avoid breaking the model forward pass
            # print(f"Warning: Prefix input hook failed for layer {self.layer_idx}: {e}")
            return input_args
    
    def _modify_attention_outputs(self, module, input_args, output):
        """Hook function to remove prefix from attention outputs."""
        if self.prefix is None:
            return output
        
        try:
            prefix_len = self.prefix.size(1)
            
            # Handle different output formats
            if isinstance(output, tuple):
                modified_outputs = []
                
                for i, item in enumerate(output):
                    if hasattr(item, 'dim') and hasattr(item, 'size') and item.dim() >= 2:
                        # This is a tensor with at least 2 dimensions
                        # Check if it has sequence length dimension that needs trimming
                        if item.size(-2) > prefix_len:  # sequence length is usually second-to-last dim
                            # Remove prefix from sequence dimension
                            if item.dim() == 3:
                                # [batch, seq_len, hidden] - main output or similar
                                modified_item = item[:, prefix_len:]
                            elif item.dim() == 4:
                                # [batch, heads, seq_len, head_dim] - key/value states
                                modified_item = item[:, :, prefix_len:]
                            else:
                                # Other dimensions - try to remove from second dimension
                                modified_item = item[:, prefix_len:]
                            modified_outputs.append(modified_item)
                        else:
                            # Dimension too small or doesn't need trimming, or already processed
                            modified_outputs.append(item)
                    elif isinstance(item, tuple):
                        # Handle nested tuples (like key/value pairs)
                        nested_modified = []
                        for subitem in item:
                            if hasattr(subitem, 'dim') and hasattr(subitem, 'size') and subitem.dim() >= 2:
                                if subitem.size(-2) > prefix_len:
                                    if subitem.dim() == 4:
                                        # [batch, heads, seq_len, head_dim] - key/value states
                                        modified_subitem = subitem[:, :, prefix_len:]
                                    elif subitem.dim() == 3:
                                        # [batch, seq_len, hidden]
                                        modified_subitem = subitem[:, prefix_len:]
                                    else:
                                        modified_subitem = subitem[:, prefix_len:]
                                    nested_modified.append(modified_subitem)
                                else:
                                    nested_modified.append(subitem)
                            else:
                                nested_modified.append(subitem)
                        modified_outputs.append(tuple(nested_modified))
                    else:
                        # Not a tensor or doesn't need modification
                        modified_outputs.append(item)
                
                return tuple(modified_outputs)
                
            elif output is not None and hasattr(output, 'dim') and output.dim() >= 2:
                # Single tensor output
                if output.size(-2) > prefix_len:
                    return output[:, prefix_len:]
                else:
                    return output
            else:
                return output
                
        except Exception as e:
            # Fail silently to avoid breaking the model forward pass
            # print(f"Warning: Prefix output hook failed for layer {self.layer_idx}: {e}")
            return output
    
    def register_hooks(self, attention_layer):
        """Register forward hooks on the attention layer."""
        # Pre-hook to modify inputs
        pre_hook = attention_layer.register_forward_pre_hook(self._modify_attention_inputs)
        # Post-hook to modify outputs  
        post_hook = attention_layer.register_forward_hook(self._modify_attention_outputs)
        
        # Store handles for cleanup
        self.hook_handle = (pre_hook, post_hook)
        return self.hook_handle
    
    def remove_hooks(self):
        """Remove the registered hooks."""
        if self.hook_handle is not None:
            pre_hook, post_hook = self.hook_handle
            pre_hook.remove()
            post_hook.remove()
            self.hook_handle = None

class TestTimePrefixModel(nn.Module):
    """A wrapper that adds test-time prefix generation to any transformer model using hooks."""
    def __init__(
        self,
        base_model: PreTrainedModel,
        config: TestTimePrefixConfig,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        initial_prefixes: Optional[torch.Tensor] = None,
        initial_states: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.tokenizer = tokenizer
        
        # Determine target layers based on selection mode
        self.target_layers = self._get_target_layers()
        print(f"Target layers to apply prefix generation to: {self.target_layers}")

        # Create prefix generator(s) for target layers
        if config.shared_weight_for_all_layers:
            # Create one shared generator that all layers will reference
            shared_generator = PrefixGenerator(config, base_model.config.hidden_size)
            self.prefix_generators = nn.ModuleDict({
                str(layer_idx): shared_generator
                for layer_idx in self.target_layers
            })
            print(f"Using shared prefix generator for all {len(self.target_layers)} layers")
        else:
            # Create individual generators for each layer
            self.prefix_generators = nn.ModuleDict({
                str(layer_idx): PrefixGenerator(config, base_model.config.hidden_size)
                for layer_idx in self.target_layers
            })
            print(f"Using individual prefix generators for {len(self.target_layers)} layers")
        
        # Initialize current prefixes and states
        if initial_prefixes is not None:
            if initial_prefixes.shape[0] != len(self.target_layers):
                raise ValueError(f"Initial prefixes must have shape [num_layers, ...], got {initial_prefixes.shape}")
            self.current_prefixes = initial_prefixes
        else:
            self.current_prefixes = None
            
        if initial_states is not None:
            if initial_states.shape[0] != len(self.target_layers):
                raise ValueError(f"Initial states must have shape [num_layers, ...], got {initial_states.shape}")
            self.current_states = initial_states
        else:
            self.current_states = None
        
        # Setup hook-based prefix handling
        self._setup_prefix_hooks()
        
        # Set initial prefixes if provided
        if self.current_prefixes is not None:
            self._set_layer_prefixes()
    
    def _setup_prefix_hooks(self):
        """Setup hook-based prefix handling."""
        self.prefix_hooks = {}
        
        for layer_idx in self.target_layers:
            # Get the attention layer
            attention_layer = self._get_attention_layer(layer_idx)
            
            # Use traditional hooks for all models
            print(f"Using traditional hooks for layer {layer_idx}")
            hook_handler = PrefixAttentionHook(layer_idx, self.base_model.config.hidden_size)
            hook_handler.register_hooks(attention_layer)
            self.prefix_hooks[layer_idx] = hook_handler
    
    def _get_attention_layer(self, layer_idx: int):
        """Get the attention layer for a given layer index."""
        if hasattr(self.base_model, 'transformer') and hasattr(self.base_model.transformer, 'h'):
            # GPT-2 style with transformer wrapper
            return self.base_model.transformer.h[layer_idx].attn
        elif hasattr(self.base_model, 'h'):
            # GPT-2 style with direct access (like DialoGPT)
            return self.base_model.h[layer_idx].attn
        elif hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'layers'):
            # Qwen CausalLM style (QwenForCausalLM has model.model.layers)
            return self.base_model.model.layers[layer_idx].self_attn
        elif hasattr(self.base_model, 'layers'):
            # Qwen base model style (and other modern architectures)
            return self.base_model.layers[layer_idx].self_attn
        elif hasattr(self.base_model, 'encoder'):
            # BERT style
            return self.base_model.encoder.layer[layer_idx].attention
        else:
            raise ValueError("Unsupported model architecture")

    def _get_target_layers(self) -> List[int]:
        """Determine which layers to apply prefix generation to."""
        total_layers = self.base_model.config.num_hidden_layers
        print(f"Total layers/transformer blocks in model: {total_layers}")

        if self.config.layer_selection_mode == "all":
            return list(range(total_layers))
        
        elif self.config.layer_selection_mode == "specific":
            if not isinstance(self.config.layer_selection, list):
                raise ValueError("layer_selection must be a list of layer indices for 'specific' mode")
            return sorted(self.config.layer_selection)
        
        elif self.config.layer_selection_mode == "exclude":
            if not isinstance(self.config.layer_selection, list):
                raise ValueError("layer_selection must be a list of layer indices for 'exclude' mode")
            return [i for i in range(total_layers) if i not in self.config.layer_selection]
        
        elif self.config.layer_selection_mode == "spread":
            if not isinstance(self.config.layer_selection, int):
                raise ValueError("layer_selection must be an integer for 'spread' mode")
            num_layers = self.config.layer_selection
            if num_layers > total_layers:
                raise ValueError(f"Cannot spread {num_layers} layers across {total_layers} total layers")
            
            # Calculate gap between layers
            gap = total_layers / num_layers
            # Start at gap/2 to center the layers
            start = int(gap / 2)
            # Generate evenly spaced indices
            return [int(start + i * gap) for i in range(num_layers)]
        
        else:
            raise ValueError(f"Unknown layer selection mode: {self.config.layer_selection_mode}")
    

    def update_prefix_given_hidden_states(self, hidden_states: List[torch.Tensor], num_iterations: Optional[int] = None) -> torch.Tensor:
        """Update prefixes using the provided hidden states as context.
        
        Args:
            hidden_states: List of hidden states for each layer [batch_size, seq_len, hidden_size]
            num_iterations: Number of iterations for prefix generation (uses config default if None)
            
        Returns:
            Updated prefixes
        """
        current_prefixes = []
        current_states = []
        
        # Simple loop - sharing happens naturally through same generator instance
        for i, layer_idx in enumerate(self.target_layers):
            # Each layer uses its own context and state
            layer_hidden_states = hidden_states[layer_idx]
            context = layer_hidden_states.to(dtype=torch.float32)  # Keep full sequence
            initial_state = self.current_states[i] if self.current_states is not None else None
            
            # Generate prefix - sharing happens because generators point to same instance
            iterations = num_iterations if num_iterations is not None else self.config.max_iterations
            # Ensure iterations is an integer
            if isinstance(iterations, torch.Tensor):
                iterations = int(iterations.item())
            elif not isinstance(iterations, int):
                iterations = int(iterations)
                
            states_iters, prefix = self.prefix_generators[str(layer_idx)](
                num_iterations=iterations,
                context=context,
                initial_state=initial_state
            )
            
            current_prefixes.append(prefix.unsqueeze(1))
            current_states.append(states_iters[:, -1])

        self.current_prefixes = torch.stack(current_prefixes, dim=0)
        self.current_states = torch.stack(current_states, dim=0)
        self._set_layer_prefixes()
        return self.current_prefixes

    def update_prefix_given_input(
        self,
        input_ids: torch.Tensor,
        num_iterations: Optional[int] = None
    ) -> torch.Tensor:
        """Update prefixes using the input tokens.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            num_iterations: Number of iterations for prefix generation (uses config default if None)
            
        Returns:
            Updated prefixes
        """
        # Get hidden states from the base model
        with torch.no_grad():
            if self.config.use_hooks_during_prefix_update:
                # Use model with prefixes (may cause recursive behavior)
                outputs = self.run_base_model_with_prefixes(
                    input_ids=input_ids,
                    output_hidden_states=True
                )
            else:
                # Use clean model without prefixes (recommended)
                outputs = self.run_base_model_without_prefixes(
                    input_ids=input_ids,
                    output_hidden_states=True
                )
            # Use all layer hidden states as context
            hidden_states = outputs.hidden_states  # List of [batch_size, seq_len, hidden_size] for each layer
        
        # Update prefixes using the hidden states
        return self.update_prefix_given_hidden_states(hidden_states, num_iterations)

    def _set_layer_prefixes(self):
        """Set prefixes for each target layer."""
        if self.current_prefixes is None:
            return
            
        for i, layer_idx in enumerate(self.target_layers):
            layer_prefix = self.current_prefixes[i]  # [batch, prefix_len, hidden]
            self.prefix_hooks[layer_idx].set_prefix(layer_prefix)

    def _clear_layer_prefixes(self):
        """Clear prefixes from all target layers without resetting current_prefixes."""
        for layer_idx in self.target_layers:
            self.prefix_hooks[layer_idx].clear_prefix()

    def run_base_model_with_prefixes(self, *args, **kwargs):
        """Run the base model with prefixes active."""
        # Ensure prefixes are set in hooks before every forward pass
        self._set_layer_prefixes()
        return self.base_model(*args, **kwargs)

    def run_base_model_without_prefixes(self, *args, **kwargs):
        """Run the base model without prefixes (clean hidden states)."""
        # Temporarily clear prefixes from hooks
        self._clear_layer_prefixes()
        try:
            return self.base_model(*args, **kwargs)
        finally:
            # Restore prefixes in hooks if they exist
            if self.current_prefixes is not None:
                self._set_layer_prefixes()

    def reset_prefixes(self):
        """Reset current prefixes and states to None."""
        self.current_prefixes = None
        self.current_states = None
        
        # Clear prefixes from hooks
        self._clear_layer_prefixes()
    
    def cleanup(self):
        """Clean up resources by removing hooks."""
        for hook_handler in self.prefix_hooks.values():
            hook_handler.remove_hooks()
        self.prefix_hooks.clear()
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except:
            pass  # Ignore cleanup errors during deletion
                
    def forward(self, *args, **kwargs):
        """Forward pass through the base model with prefixes."""
        return self.run_base_model_with_prefixes(*args, **kwargs)
        
    def generate(self, *args, **kwargs):
        """Generate text using the base model with prefixes."""
        # Ensure prefixes are set in hooks
        self._set_layer_prefixes()
        return self.base_model.generate(*args, **kwargs)

def create_test_time_prefix_config(
    layer_selection_mode: str = "all",
    layer_selection: Optional[Union[List[int], int]] = None,
    hidden_size: Optional[int] = None,
    max_iterations: int = 10,
    gradient_steps: int = 4,
    shared_weight_for_all_layers: bool = False,
    use_hooks_during_prefix_update: bool = False
) -> TestTimePrefixConfig:
    """Create a test-time prefix configuration.
    
    Args:
        layer_selection_mode: How to select layers ("all", "specific", "exclude", "spread")
        layer_selection: List of layer indices or number of layers to spread
        hidden_size: Hidden size for prefix generator (uses model's if None)
        max_iterations: Maximum iterations for prefix generation
        gradient_steps: Number of iterations to allow gradients through
        shared_weight_for_all_layers: Whether all layers share the same prefix generator
        use_hooks_during_prefix_update: Whether to use hooks during prefix generation
        
    Returns:
        TestTimePrefixConfig object
    """
    return TestTimePrefixConfig(
        layer_selection_mode=layer_selection_mode,
        layer_selection=layer_selection,
        hidden_size=hidden_size,
        max_iterations=max_iterations,
        gradient_steps=gradient_steps,
        shared_weight_for_all_layers=shared_weight_for_all_layers,
        use_hooks_during_prefix_update=use_hooks_during_prefix_update
    )

def add_instance_level_prefix_generator(
    model: PreTrainedModel,
    config: Optional[TestTimePrefixConfig] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None
) -> TestTimePrefixModel:
    """Decorator function to add instance-level prefix generation to a model.
    
    Args:
        model: The base transformer model
        config: Configuration for prefix generation
        tokenizer: Optional tokenizer
    """
    if config is None:
        config = create_test_time_prefix_config(
            hidden_size=model.config.hidden_size
        )
    return TestTimePrefixModel(model, config, tokenizer)

 