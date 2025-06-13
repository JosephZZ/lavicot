import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, Callable
import numpy as np


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
    def __init__(self, config, model_hidden_size: int):
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
