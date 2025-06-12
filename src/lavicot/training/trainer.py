import os
import sys
import datetime
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from datasets import load_dataset
from tqdm import tqdm
import random
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
import time
import wandb
from torch.nn.utils import clip_grad_norm_
from types import SimpleNamespace


from ..evaluation.evaluator import evaluate_model_configurations
from ..utils.logging_utils import (
    get_gpu_memory_usage, get_gradient_norm, get_parameter_norm,
    get_token_statistics, initialize_wandb_tracking, print_training_configuration
)
from ..utils.data_utils import prepare_batch, prepare_datasets_and_samples, extract_gsm8k_data_components
from .training_utils import setup_training_environment, compute_weighted_loss
from .model_setup_utils import setup_model_and_training_components
from .checkpoint_utils import save_checkpoint
from .components.handlers import PrefixUpdateHandler, LossComputeHandler, OptimizationHandler


class LaviCotTrainer:
    """LaviCot Trainer class that encapsulates training logic."""
    
    def __init__(self, config: SimpleNamespace):
        """Initialize the trainer with configuration."""
        self.config = config
        self.device = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.tokenizer = None
        
        # Training state
        self.start_epoch = 0
        self.start_global_step = 0
        self.best_accuracy = 0.0
        self.global_step = 0
        
        # Data
        self.train_indices = None
        self.eval_instances = None
        self.full_eval_instances = None
        self.train_dataset = None
        self.test_dataset = None
        
        # Training components
        self.prefix_handler = None
        self.loss_handler = None
        self.optimization_handler = None
        
        
    def setup(self):
        """Setup all training components."""
        # 1. Setup training environment
        self.device = setup_training_environment(self.config)
        
        # 2. Prepare datasets and samples
        (self.train_indices, self.eval_instances, self.full_eval_instances, 
         self.train_dataset, self.test_dataset) = prepare_datasets_and_samples(self.config)
        

        # 3. Setup model, optimizer, scheduler and tokenizer
        (self.model, self.optimizer, self.scheduler, self.tokenizer, 
         self.start_epoch, self.start_global_step, self.best_accuracy) = setup_model_and_training_components(
            self.config, self.device
        )
        
        # 4. Initialize WandB tracking
        initialize_wandb_tracking(self.config, self.model)
        
        # 5. Print training configuration
        print_training_configuration(
            self.config, self.train_indices, self.eval_instances, 
            self.full_eval_instances, self.start_global_step
        )
        
        # Initialize global step
        self.global_step = self.start_global_step
        
        # Initialize training component handlers
        self.prefix_handler = PrefixUpdateHandler(self.config, self.model, self.tokenizer, self.device)
        self.loss_handler = LossComputeHandler(self.config, self.model, self.tokenizer, self.device)
        self.optimization_handler = OptimizationHandler(self.config, self.model, self.optimizer, self.scheduler)
        
    def train(self):
        """Main training loop."""
        self.setup()
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.config.num_train_epochs):
            self._train_epoch(epoch)
            
        self._final_evaluation()
        self._save_final_checkpoint()
        
        if self.config.use_wandb:
            wandb.finish()
        
        time_used = time.time() - start_time
        print(f"Training completed in {int(time_used // 3600)}h {int((time_used % 3600) // 60)}m {int(time_used % 60)}s")
        
    def _train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        
        # Shuffle training data indices
        shuffled_train_indices = self.train_indices.copy()
        random.shuffle(shuffled_train_indices)
        
        batch_iterator = self._create_batch_iterator(shuffled_train_indices, epoch)
        
        for batch_instances in batch_iterator:
            self._train_step(batch_instances)
            
    def _create_batch_iterator(self, shuffled_train_indices: List[int], epoch: int):
        """Create an iterator over training batches."""
        batch_size = self.config.per_device_train_batch_size
        progress_bar = tqdm(
            range(0, len(shuffled_train_indices), batch_size), 
            desc=f"Epoch {epoch + 1}"
        )
        
        for i in progress_bar:
            batch_train_indices = shuffled_train_indices[i:i + batch_size]
            batch_instances = [self.train_dataset[idx] for idx in batch_train_indices]
            yield batch_instances
            
    def _train_step(self, batch_instances: List[Dict]):
        """Execute one training step with a batch."""
        # 1. Update prefix with multiple rounds
        self.prefix_handler.update_prefix_multi_round(batch_instances)
        
        # 2. Compute loss
        loss = self.loss_handler.compute_loss(batch_instances)
        
        # 3. Backward pass and optimization
        gradient_norm = self.optimization_handler.backward_and_optimize(loss, self.global_step)
        
        # 4. Log metrics if gradients were updated
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            self._log_training_metrics(loss, gradient_norm)
        
        # 5. Evaluate if needed
        if (self.global_step + 1) % self.config.eval_steps == 0:
            self._evaluate_during_training()
            
        # 6. Save checkpoint if needed
        if (self.global_step + 1) % getattr(self.config, 'save_steps', 1000) == 0:
            self._save_checkpoint()
            
        self.global_step += 1
        

            
    def _log_training_metrics(self, loss: torch.Tensor, gradient_norm: float):
        """Log training metrics to wandb."""
        if self.config.use_wandb:
            current_rounds = self.prefix_handler._get_current_rounds()
            wandb.log({
                "train/loss": loss.item(),
                "train/learning_rate": self.scheduler.get_last_lr()[0],
                "train/gradient_norm": gradient_norm,
                "train/parameter_norm": get_parameter_norm(self.model),
                "train/gpu_memory": get_gpu_memory_usage(),
                "train/rounds": current_rounds,
                "train/global_step": self.global_step
            })
            
    def _evaluate_during_training(self):
        """Perform evaluation during training."""
        self.model.eval()
        with torch.no_grad():
            eval_results, eval_outputs = self._run_evaluation(
                self.eval_instances[:self.config.eval_during_training_samples]
            )
            
            # Evaluate prefix iteration loss curves
            prefix_loss_results = self._evaluate_prefix_loss_curves()
            
            # Save evaluation outputs
            self._save_evaluation_outputs(eval_results, eval_outputs, prefix_loss_results)
            
            # Update best accuracy
            current_accuracy = list(eval_results.values())[0].get('accuracy', 0.0)
            if current_accuracy > self.best_accuracy:
                self.best_accuracy = current_accuracy
                
            # Log to wandb
            self._log_evaluation_metrics(eval_results, prefix_loss_results)
            
        self.model.train()
        
    def _run_evaluation(self, eval_instances: List[Dict]) -> Tuple[Dict, List]:
        """Run model evaluation on given instances."""
        return evaluate_model_configurations(
            self.model, self.tokenizer, eval_instances, self.config.dataset_name,
            self.device, self.config.max_length, self.config.evaluation_settings,
            temperature=getattr(self.config, 'generation_temperature', 0.7),
            extract_number_from_answer_only=getattr(self.config, 'extract_number_from_answer_only', False)
        )
        
    def _evaluate_prefix_loss_curves(self) -> List[Dict]:
        """Evaluate prefix iteration loss curves."""
        from ..evaluation.evaluator import evaluate_batch_sequence_prediction_loss
        
        return evaluate_batch_sequence_prediction_loss(
            self.model, self.tokenizer,
            self.eval_instances[:self.config.eval_during_training_samples],
            self.config.dataset_name,
            self.device, self.config.max_length,
            prefix_iterations_range=(1, self.config.prefix_generator['max_iterations'])
        )
        
    def _save_evaluation_outputs(self, eval_results: Dict, eval_outputs: List, prefix_loss_results: List[Dict]):
        """Save evaluation outputs to file."""
        os.makedirs(os.path.join(self.config.output_dir, "eval_outputs"), exist_ok=True)
        eval_outputs_file = os.path.join(
            self.config.output_dir, "eval_outputs", f"step_{self.global_step + 1}.json"
        )
        
        with open(eval_outputs_file, "w") as f:
            json.dump({
                "global_step": self.global_step + 1,
                "epoch": self._current_epoch,
                "evaluation_type": "during_training",
                "num_samples": self.config.eval_during_training_samples,
                "results": eval_results,
                "model_outputs": eval_outputs,
                "prefix_loss_curves": prefix_loss_results,
                "config": {
                    "generation_temperature": getattr(self.config, 'generation_temperature', 0.7),
                    "evaluation_settings": self.config.evaluation_settings
                }
            }, f, indent=2)
            
        print(f"Saved evaluation outputs to {eval_outputs_file}")
        
    def _log_evaluation_metrics(self, eval_results: Dict, prefix_loss_results: List[Dict]):
        """Log evaluation metrics to wandb."""
        if not self.config.use_wandb:
            return
            
        # Log evaluation results
        for setting_name, metrics in eval_results.items():
            for metric_name, value in metrics.items():
                wandb.log({
                    f"eval/{setting_name}/{metric_name}": value,
                    "train/global_step": self.global_step
                })
                
        # Log prefix loss curves
        self._log_prefix_loss_curves(prefix_loss_results)
        
    def _log_prefix_loss_curves(self, prefix_loss_results: List[Dict]):
        """Log prefix loss curves to wandb."""
        if not prefix_loss_results:
            return
            
        all_iterations = list(prefix_loss_results[0].keys())
        all_ys = []
        all_keys = []
        
        for idx, loss_dict in enumerate(prefix_loss_results):
            losses = [loss_dict[iter_count] for iter_count in all_iterations]
            all_ys.append(losses)
            all_keys.append(f"Instance {idx}")
            
        # Log current evaluation curves
        wandb.log({
            "prefix_loss/current_eval_all_instances": wandb.plot.line_series(
                xs=all_iterations,
                ys=all_ys,
                keys=all_keys,
                title=f"Current Eval: All Instance Loss Curves (Step {self.global_step + 1})",
                xname="Prefix Iterations"
            )
        })
        
        # Log training progress for specific iterations
        self._log_prefix_iteration_progress(prefix_loss_results, all_iterations)
        
        # Log average loss curves
        self._log_average_loss_curves(prefix_loss_results, all_iterations)
        
    def _log_prefix_iteration_progress(self, prefix_loss_results: List[Dict], all_iterations: List[int]):
        """Log prefix iteration progress for specific iterations."""
        tracked_iterations = [1, 3, 7]
        
        for iter_count in tracked_iterations:
            if iter_count in all_iterations:
                valid_losses = [
                    result[iter_count] for result in prefix_loss_results 
                    if iter_count in result and result[iter_count] != float('inf')
                ]
                if valid_losses:
                    avg_loss = sum(valid_losses) / len(valid_losses)
                    wandb.log({
                        f"prefix_loss_progress/iter_{iter_count}": avg_loss,
                        "train/global_step": self.global_step
                    })
                    
    def _log_average_loss_curves(self, prefix_loss_results: List[Dict], all_iterations: List[int]):
        """Log average loss curves across evaluation steps."""
        # Compute average loss across instances for each iteration
        avg_losses_by_iteration = []
        for iter_count in all_iterations:
            valid_losses = [
                result[iter_count] for result in prefix_loss_results 
                if iter_count in result and result[iter_count] != float('inf')
            ]
            if valid_losses:
                avg_loss = sum(valid_losses) / len(valid_losses)
                avg_losses_by_iteration.append(avg_loss)
            else:
                avg_losses_by_iteration.append(float('inf'))
                
        # Store cumulative curves data
        if not hasattr(evaluate_model_configurations, '_avg_curve_data'):
            evaluate_model_configurations._avg_curve_data = {
                'xs': all_iterations,
                'ys': [],
                'keys': []
            }
            
        # Add current curve
        evaluate_model_configurations._avg_curve_data['ys'].append(avg_losses_by_iteration)
        evaluate_model_configurations._avg_curve_data['keys'].append(f"Step {self.global_step + 1}")
        
        # Log the cumulative plot
        wandb.log({
            "prefix_loss/average_curves_by_step": wandb.plot.line_series(
                xs=evaluate_model_configurations._avg_curve_data['xs'],
                ys=evaluate_model_configurations._avg_curve_data['ys'],
                keys=evaluate_model_configurations._avg_curve_data['keys'],
                title="Average Loss Curves Across Training Steps",
                xname="Prefix Iterations"
            )
        })
        
    def _save_checkpoint(self):
        """Save training checkpoint."""
        print(f"Saving checkpoint at step {self.global_step + 1}")
        save_checkpoint(
            self.model, self.optimizer, self._current_epoch, 
            self.global_step + 1, self.best_accuracy,
            self.config.output_dir, self.global_step + 1
        )
        
    def _final_evaluation(self):
        """Run final evaluation on full test set."""
        print("\nRunning final evaluation on full test set...")
        self.model.eval()
        
        with torch.no_grad():
            final_eval_results, final_model_outputs = self._run_evaluation(self.full_eval_instances)
            
        # Log final results
        if self.config.use_wandb:
            for config_name, metrics in final_eval_results.items():
                for metric_name, value in metrics.items():
                    wandb.log({
                        f"final_eval/{config_name}/{metric_name}": value
                    })
                    
        # Save final evaluation results
        self._save_final_evaluation_results(final_eval_results, final_model_outputs)
        
        # Print final summary
        print("\nFinal Evaluation Summary:")
        for setting_name, metrics in final_eval_results.items():
            print(f"{setting_name}: {metrics['accuracy']:.2f}%")
            
    def _save_final_evaluation_results(self, final_eval_results: Dict, final_model_outputs: List):
        """Save final evaluation results to file."""
        with open(os.path.join(self.config.output_dir, "final_evaluation_results.json"), "w") as f:
            json.dump({
                "global_step": self.global_step,
                "total_epochs": self.config.num_train_epochs,
                "evaluation_type": "final",
                "num_samples": len(self.full_eval_instances),
                "results": final_eval_results,
                "model_outputs": final_model_outputs,
                "training_config": {
                    "generation_temperature": getattr(self.config, 'generation_temperature', 0.7),
                    "evaluation_settings": self.config.evaluation_settings,
                    "model_name": self.config.model_name,
                    "dataset_name": self.config.dataset_name
                }
            }, f, indent=2)
            
    def _save_final_checkpoint(self):
        """Save final checkpoint and model."""
        print("Saving final checkpoint...")
        save_checkpoint(
            self.model, self.optimizer, self.config.num_train_epochs, 
            self.global_step, self.best_accuracy, self.config.output_dir, self.global_step
        )
        
        # Save final model
        torch.save(
            self.model.state_dict(),
            os.path.join(self.config.output_dir, "final_model.pt")
        )
        
    @property
    def _current_epoch(self) -> int:
        """Get current epoch based on global step."""
        # This is an approximation - you might want to track epoch more precisely
        steps_per_epoch = len(self.train_indices) // self.config.per_device_train_batch_size
        return self.start_epoch + (self.global_step // steps_per_epoch) 