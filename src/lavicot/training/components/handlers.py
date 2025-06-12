"""Training components for prefix generation and round handling."""

import random
from typing import List, Dict, Tuple
import torch
from transformers import PreTrainedTokenizer

from ...utils.data_utils import prepare_batch


class PrefixUpdateHandler:
    """Handles prefix update logic for multi-round training."""
    
    def __init__(self, config, model, tokenizer: PreTrainedTokenizer, device):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def update_prefix_multi_round(self, batch_instances: List[Dict]):
        """Update model prefix through multiple rounds."""
        self.model.reset_prefixes()
        
        # Determine number of rounds for this batch
        current_rounds = self._get_current_rounds()
        
        for round_idx in range(current_rounds):
            self._update_prefix_single_round(batch_instances, round_idx, current_rounds)
            
    def _get_current_rounds(self) -> int:
        """Determine the number of rounds for the current batch."""
        if self.config.stochastic_rounds:
            return random.randint(1, self.config.num_rounds)
        else:
            return self.config.num_rounds
            
    def _update_prefix_single_round(self, batch_instances: List[Dict], round_idx: int, current_rounds: int):
        """Update prefix for a single round."""
        prep_mode, min_prop, max_prop = self._get_round_parameters(round_idx, current_rounds)
        
        # Prepare input for this round
        batch_inputs = prepare_batch(
            prep_mode=prep_mode,
            data_instances=batch_instances,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
            min_proportion=min_prop,
            max_proportion=max_prop,
            device=self.device
        )
        
        # Update prefix
        num_iterations = random.randint(1, self.config.prefix_generator['max_iterations'])
        self.model.update_prefix_given_input(
            input_ids=batch_inputs,
            num_iterations=num_iterations
        )
        
    def _get_round_parameters(self, round_idx: int, current_rounds: int) -> Tuple[str, float, float]:
        """Get parameters for a specific round."""
        if round_idx == 0:
            # First round: question only
            return "question_only", 0.0, 0.0
        else:
            # Later rounds: partial reasoning
            return self._calculate_cot_proportions(round_idx, current_rounds)
            
    def _calculate_cot_proportions(self, round_idx: int, current_rounds: int) -> Tuple[str, float, float]:
        """Calculate proportion range for CoT rounds."""
        prep_mode = "cot_only"
        
        if current_rounds == 2:
            # Only one more round after question-only: use config min/max
            min_prop = getattr(self.config, 'proportion_min', 0.3)
            max_prop = getattr(self.config, 'proportion_max', 0.7)
        else:
            # Multiple rounds: split range into (current_rounds-1) parts
            proportion_min = getattr(self.config, 'proportion_min', 0.3)
            proportion_max = getattr(self.config, 'proportion_max', 0.7)
            
            # Split range into (current_rounds-1) segments
            num_segments = current_rounds - 1
            segment_size = (proportion_max - proportion_min) / num_segments
            segment_idx = round_idx - 1  # 0-indexed for segments (skip question-only round)
            
            min_prop = proportion_min + segment_idx * segment_size
            max_prop = proportion_min + (segment_idx + 1) * segment_size
            
        return prep_mode, min_prop, max_prop


class LossComputeHandler:
    """Handles loss computation for training."""
    
    def __init__(self, config, model, tokenizer: PreTrainedTokenizer, device):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def compute_loss(self, batch_instances: List[Dict]) -> torch.Tensor:
        """Compute loss for the batch."""
        from ..training_utils import compute_weighted_loss
        
        # Prepare full sequences for output generation and get question lengths
        full_sequences, question_token_lengths = prepare_batch(
            prep_mode="full",
            data_instances=batch_instances,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
            min_proportion=1.0,
            max_proportion=1.0,
            device=self.device,
            return_question_token_lengths=True
        )
        
        # Generate output with full sequences
        outputs = self.model(
            input_ids=full_sequences,
            attention_mask=torch.ones_like(full_sequences),
            labels=full_sequences
        )
        
        # Compute weighted loss
        loss = compute_weighted_loss(
            outputs.logits,
            full_sequences,
            question_token_lengths,
            self.config.seen_token_weight,
            self.config.unseen_token_weight
        )
        
        # Normalize loss by gradient accumulation steps
        return loss / self.config.gradient_accumulation_steps


class OptimizationHandler:
    """Handles optimization and gradient updates."""
    
    def __init__(self, config, model, optimizer, scheduler):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        
    def backward_and_optimize(self, loss: torch.Tensor, global_step: int) -> float:
        """Perform backward pass and optimization. Returns gradient norm if updated."""
        from ...utils.logging_utils import get_gradient_norm
        from torch.nn.utils import clip_grad_norm_
        
        loss.backward()
        gradient_norm = 0.0
        
        # Update weights if we've accumulated enough gradients
        if (global_step + 1) % self.config.gradient_accumulation_steps == 0:
            # Calculate gradient norm before clipping
            gradient_norm = get_gradient_norm(self.model, self.model.prefix_generators.parameters())
            
            # Clip gradients - only for prefix generator parameters
            clip_grad_norm_(self.model.prefix_generators.parameters(), self.config.max_grad_norm)
            
            # Update weights
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
        return gradient_norm 