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


def parse_config_overrides(unknown_args):
    """Parse unknown arguments as config overrides.
    
    Args:
        unknown_args: List of unknown command line arguments
        
    Returns:
        Dict of config overrides
    """
    overrides = {}
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if arg.startswith('--'):
            key = arg[2:]  # Remove '--' prefix
            if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith('--'):
                # Next argument is the value
                value = unknown_args[i + 1]
                # Try to convert to appropriate type
                try:
                    # Handle boolean strings first
                    if value.lower() in ('true', 'false'):
                        value = value.lower() == 'true'
                    # Try int if no decimal point and not a boolean
                    elif '.' not in value and value.lstrip('-').isdigit():
                        value = int(value)
                    # Try float if it has decimal point or scientific notation
                    elif ('.' in value or 'e' in value.lower()) and value.replace('.', '').replace('-', '').replace('+', '').replace('e', '').isdigit():
                        value = float(value)
                    # Handle special string values
                    elif value.lower() == 'none':
                        value = None
                    # Otherwise keep as string, removing quotes if present
                    else:
                        value = value.strip('"\'')
                except ValueError:
                    # If all conversions fail, keep as string
                    value = value.strip('"\'')
                
                overrides[key] = value
                i += 2
            else:
                # Flag without value, treat as True
                overrides[key] = True
                i += 1
        else:
            i += 1
    
    return overrides


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Train LaViCoT model")
    parser.add_argument("--config", type=str, default='./src/lavicot/config/defaults/default.yaml',
                       help="Path to configuration file")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Dataset configuration name (e.g., 'gsm8k', 'math')")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Path to checkpoint folder or file to resume from")
    parser.add_argument("--auto_resume", action="store_true", default=False,
                       help="Automatically resume from latest checkpoint in output_dir")
    parser.add_argument("--resume_config_too", action="store_true", default=True,
                       help="When resuming, also load the config from checkpoint folder")
    parser.add_argument("--base_output_dir", type=str, default='./outputs/debug',
                       help="Output directory to save checkpoints and results")
    
    # Parse known and unknown arguments
    args, unknown_args = parser.parse_known_args()
    
    # Parse config overrides from unknown arguments
    config_overrides = parse_config_overrides(unknown_args)
    if config_overrides:
        print(f"Config overrides from command line: {config_overrides}")
    
    # Setup configuration and paths using config_utils
    config = setup_config_and_paths(args, config_overrides)
    if config is None:
        print("Error: Failed to setup configuration")
        return
    
    # Create trainer and train
    trainer = LaviCotTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main() 