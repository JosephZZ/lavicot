#!/usr/bin/env python3
"""Main evaluation script for LaViCoT."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
from src.lavicot.evaluation.evaluator import evaluate
from src.lavicot.config.config_loader import load_config


def main():
    """Main entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate LaViCoT model")
    parser.add_argument("--config", type=str, 
                       help="Path to configuration file")
    parser.add_argument("--model_name", type=str, default="distilgpt2",
                       help="Model name or path")
    parser.add_argument("--checkpoint_path", type=str,
                       help="Path to model checkpoint")
    parser.add_argument("--num_eval_samples", type=int, default=100,
                       help="Number of evaluation samples")
    parser.add_argument("--output_dir", type=str, 
                       help="Output directory for results")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluate(
        model_name=args.model_name,
        config_path=args.config,
        checkpoint_path=args.checkpoint_path,
        num_eval_samples=args.num_eval_samples,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb
    )


if __name__ == "__main__":
    main() 