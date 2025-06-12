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
    
    # Wrapper control configuration  
    use_hooks_during_prefix_update: bool = False  # Whether to use prefixes during prefix generation (legacy name)

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



class GenericPrefixAttentionWrapper:
    """Generic wrapper for attention layers that handles prefix injection via monkey patching.
    
    Works with different model architectures by automatically detecting the forward signature.
    """
    
    def __init__(self, original_attention, layer_idx: int, parent_model):
        self.original_attention = original_attention
        self.layer_idx = layer_idx
        self.parent_model = parent_model
        self.prefix = None
        
        # Store original forward method
        self.original_forward = original_attention.forward
        
        # Detect model architecture for signature handling
        self.model_type = self._detect_model_type()
        
        # Replace forward method with our wrapped version
        original_attention.forward = self.wrapped_forward
    
    def _detect_model_type(self):
        """Detect the model architecture based on attention layer type."""
        attention_type = type(self.original_attention).__name__
        
        if 'Qwen' in attention_type:
            return 'qwen'
        elif 'GPT2' in attention_type or 'Attention' in attention_type:
            return 'gpt2'
        elif 'BertAttention' in attention_type:
            return 'bert'
        else:
            # Default to trying Qwen signature first, then GPT-2
            return 'auto'
    
    def set_prefix(self, prefix: torch.Tensor):
        """Set the prefix for this layer."""
        self.prefix = prefix
        
    def clear_prefix(self):
        """Clear the prefix for this layer."""
        self.prefix = None
    
    def wrapped_forward(self, *args, **kwargs):
        """
        Wrapper for attention forward that applies prefix as bias to output hidden states.
        
        This method intercepts the forward call to any attention layer and:
        1. Calls the original attention forward method
        2. Adds prefix bias to the output hidden states
        
        Supports both traditional positional arguments and modern keyword-only arguments.
        The bias approach is much simpler than concatenation as it doesn't change sequence length.
        """
        
        # First, call the original forward method to get the output
        output = self.original_forward(*args, **kwargs)
        
        # Apply prefix as bias to the output if present
        if self.prefix is not None:
            # Convert prefix to match output's dtype and device
            if isinstance(output, tuple):
                reference_tensor = output[0]
            else:
                reference_tensor = output
                
            prefix_bias = self.prefix.to(
                dtype=reference_tensor.dtype, 
                device=reference_tensor.device
            )
            
            # Calculate norms for rescaling
            # Get hidden states (first element of output)
            hidden_states = reference_tensor
            
            # Calculate norm for each position: [batch_size, seq_len, 1]
            hidden_states_norm = torch.norm(hidden_states, dim=-1, keepdim=True).mean()  # [4, 384, 1]
            
            # Calculate norm of prefix bias: [batch_size, 1, 1]
            prefix_bias_norm = torch.norm(prefix_bias, dim=-1, keepdim=True).mean()  # [batch_size, 1, 1]
            
            # Rescale prefix bias to be 0.03 times the mean hidden states norm
            target_norm = 0.03 * hidden_states_norm  # scalar (mean across all positions)
            
            # Calculate scale factor (avoid division by zero)
            if prefix_bias_norm.item() > 1e-8:
                scale_factor = target_norm / prefix_bias_norm.item()  # scalar
            else:
                scale_factor = 0.01
            
            # Add prefix bias to output
            # assuming prefix_bias shape: [batch_size, 1, hidden_size]
            # Apply bias directly to output
            if isinstance(output, tuple):
                # Modify first element (hidden_states) and keep other outputs unchanged
                output = (output[0] * (1-scale_factor) + prefix_bias * scale_factor,) + output[1:]
            else:
                # Direct tensor output
                output = output * (1-scale_factor) + prefix_bias * scale_factor
        
        return output
    
    def restore_original(self):
        """Restore the original forward method."""
        self.original_attention.forward = self.original_forward

class TestTimePrefixModel(nn.Module):
    """A wrapper that adds test-time prefix generation to any transformer model using wrappers."""
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
        
        # Setup wrapper-based prefix handling
        self._setup_prefix_wrappers()
        
        # Set initial prefixes if provided
        if self.current_prefixes is not None:
            self._set_layer_prefixes()
    
    def _setup_prefix_wrappers(self):
        """Setup wrapper-based prefix handling for all model architectures."""
        self.prefix_wrappers = {}
        
        for layer_idx in self.target_layers:
            # Get the attention layer
            attention_layer = self._get_attention_layer(layer_idx)
            
            # Use generic wrapper for all models
            print(f"Using generic wrapper for layer {layer_idx}")
            wrapper = GenericPrefixAttentionWrapper(attention_layer, layer_idx, self)
            self.prefix_wrappers[layer_idx] = wrapper
    
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
            gap = int(total_layers / num_layers)
            # apply layers every gap layers backward from the last layer
            return [int(total_layers-1 - i*gap) for i in range(num_layers)]
        
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
            self.prefix_wrappers[layer_idx].set_prefix(layer_prefix)

    def _clear_layer_prefixes(self):
        """Clear prefixes from all target layers without resetting current_prefixes."""
        for layer_idx in self.target_layers:
            self.prefix_wrappers[layer_idx].clear_prefix()

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
        
        # Clear prefixes from wrappers
        self._clear_layer_prefixes()
    
    def set_zero_prefixes(self):
        """Set all current prefixes to zero values for testing purposes."""
        if self.current_prefixes is not None:
            # Zero out existing prefixes
            self.current_prefixes.zero_()
            
            # Apply zeroed prefixes to all wrapper layers
            self._set_layer_prefixes()
            
            print(f"Set zero prefixes: {self.current_prefixes.shape}")
        else:
            print("No current prefixes to zero out. Use update_prefix_given_input() first to create prefixes.")
    
    def cleanup(self):
        """Clean up resources by restoring original forward methods."""
        for wrapper in self.prefix_wrappers.values():
            wrapper.restore_original()
        self.prefix_wrappers.clear()
    
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
        # Ensure prefixes are set in wrappers
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
        use_hooks_during_prefix_update: Whether to use prefixes during prefix generation (legacy parameter name)
        
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