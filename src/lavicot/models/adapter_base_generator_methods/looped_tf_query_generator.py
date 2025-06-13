import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, Callable
import numpy as np
from dataclasses import dataclass

@dataclass
class TransformerPrefixConfig:
    """Configuration for transformer-based prefix generation."""
    # Query configuration
    num_queries: int = 1  # Number of query tokens to generate
    hidden_size: int = 768  # Hidden size for queries and attention
    num_blocks: int = 1  # Number of transformer blocks
    num_heads: int = 12  # Number of attention heads
    dropout: float = 0.1  # Dropout rate
    use_learnable_queries: bool = False  # Whether to use learnable queries instead of noise
    
    # Training configuration
    gradient_steps: int = 4  # Number of iterations to allow gradients through

class TransformerSelfCrossAttenBlock(nn.Module):
    """A single transformer block with self-attention and cross-attention."""
    def __init__(self, config: TransformerPrefixConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, 4 * config.hidden_size),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(4 * config.hidden_size, config.hidden_size)
        )
        
        # Layer normalizations
        self.norm1 = nn.RMSNorm(config.hidden_size, elementwise_affine=False)  # Pre-norm for self-attention
        self.norm2 = nn.RMSNorm(config.hidden_size, elementwise_affine=False)  # Pre-norm for cross-attention
        self.norm3 = nn.RMSNorm(config.hidden_size, elementwise_affine=False)  # Pre-norm for FFN
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through the transformer block.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            context: Context tensor for cross-attention [batch_size, context_len, hidden_size]
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        # Self-attention with pre-norm
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x)
        x = self.dropout(x)
        x = residual + x
        
        # Cross-attention with pre-norm
        if context is not None:
            residual = x
            x = self.norm2(x)
            x, _ = self.cross_attn(x, context, context)
            x = self.dropout(x)
            x = residual + x
        
        # Feed-forward network with pre-norm
        residual = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + x
        
        return x

class LoopedTransformerQueryGenerator(nn.Module):
    """Prefix generator using transformer blocks for self-attention between queries and cross-attention with context."""
    def __init__(self, config: TransformerPrefixConfig, num_queries, model_hidden_size: int, model_num_heads: int):
        super().__init__()
        self.config = config
        self.config.num_heads = model_num_heads
        self.config.hidden_size = model_hidden_size
        self.config.num_queries = num_queries
        # Initialize queries
        if config.use_learnable_queries:
            self.queries = nn.Parameter(
                torch.randn(1, config.num_queries, config.hidden_size) / np.sqrt(self.config.hidden_size)
            )
        else:
            self.queries = None
            
        # Create transformer blocks
        self.blocks = nn.ModuleList([
            TransformerSelfCrossAttenBlock(config)
            for _ in range(config.num_blocks)
        ])
        
        self.output_norm = nn.RMSNorm(model_hidden_size, elementwise_affine=False)  # Final norm after projection
        self.output_projection = nn.Linear(config.hidden_size, model_hidden_size)

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
            initial_state: Optional initial state tensor [batch_size, num_queries, hidden_size]
            
        Returns:
            Tuple of (outputs, final_state)
            - outputs: [batch_size, num_iterations, num_queries, hidden_size]
            - final_state: [batch_size, num_queries, hidden_size]
        """
        batch_size = context.shape[0]
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        
        # Initialize queries
        if self.config.use_learnable_queries:
            queries = self.queries.expand(batch_size, -1, -1)
        else:
            # Use principled initialization scaled by sqrt(hidden_size)
            init_std = 1.0 / np.sqrt(self.config.hidden_size)
            queries = torch.randn(
                batch_size,
                self.config.num_queries,
                self.config.hidden_size,
                device=device,
                dtype=dtype
            ) * init_std
            
        if initial_state is not None:
            queries = initial_state.to(device=device, dtype=dtype)
        
        # Process through iterations
        states_iters = []
        for i in range(num_iterations):
            # Process through transformer blocks
            for block in self.blocks:
                queries = block(queries, context)
            # Store state for this iteration
            states_iters.append(queries)

            # Only keep gradients for the last gradient_steps iterations
            if i < num_iterations - self.config.gradient_steps:
                queries = queries.detach()
        
        # Stack outputs
        states_iters = torch.stack(states_iters, dim=1)  # [batch_size, num_iterations, num_queries, hidden_size]

        # query output norm and projection
        outputs = self.output_projection(self.output_norm(queries))
        return states_iters, outputs 