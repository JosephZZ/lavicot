#!/usr/bin/env python3
"""Main training script for LaViCoT."""

import argparse
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.lavicot.training.trainer import LaviCotTrainer
from src.lavicot.training.config_utils import setup_config_and_paths


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Train LaViCoT model")
    parser.add_argument("--config", type=str, default='./src/lavicot/config/defaults/debug.yaml',
                       help="Path to configuration file")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Dataset configuration name (e.g., 'gsm8k', 'math')")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Path to checkpoint folder or file to resume from")
    parser.add_argument("--auto_resume", action="store_true", default=False,
                       help="Automatically resume from latest checkpoint in output_dir")
    parser.add_argument("--resume_config_too", action="store_true", default=True,
                       help="When resuming, also load the config from checkpoint folder")
    parser.add_argument("--output_dir_parent", type=str, default='./outputs/debug',
                       help="Output directory to save checkpoints and results")
    
    args = parser.parse_args()
    
    # Setup configuration and paths using config_utils
    config = setup_config_and_paths(args)
    if config is None:
        print("Error: Failed to setup configuration")
        return
    
    # Create trainer and train
    trainer = LaviCotTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main() 