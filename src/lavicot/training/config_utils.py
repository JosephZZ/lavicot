import os
import datetime
import argparse
from types import SimpleNamespace
from typing import Optional

from ..config.config_loader import load_config, save_config
from .checkpoint_utils import find_latest_checkpoint, find_latest_checkpoint_folder


def setup_config_and_paths(args) -> SimpleNamespace:
    """Setup configuration and paths based on command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Configured config object with resume_checkpoint_path and output_dir set
    """
    # Load initial config
    config = load_config(args.config) 
    
    # Initialize parameters to set up
    resume_checkpoint_path = None
    output_dir = None
    
    # 1. Handle auto_resume: find latest folder and checkpoint
    if args.auto_resume:
        print("Auto-resume enabled: looking for latest checkpoint...")
        base_output_dir = args.output_dir_parent
        latest_folder = find_latest_checkpoint_folder(base_output_dir)
        if latest_folder:
            latest_checkpoint = find_latest_checkpoint(latest_folder)
            if latest_checkpoint:
                resume_checkpoint_path = latest_checkpoint
                output_dir = latest_folder
                print(f"Auto-resume: found checkpoint {resume_checkpoint_path}")
            else:
                print(f"Auto-resume: no checkpoint files found in {latest_folder}")
        else:
            print(f"Auto-resume: no checkpoint folders found in {base_output_dir}")
    
    # 2. Handle explicit checkpoint path
    elif args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint
        print(f"Explicit resume from: {checkpoint_path}")
        
        if os.path.isfile(checkpoint_path):
            # It's a specific checkpoint file
            resume_checkpoint_path = checkpoint_path
            output_dir = os.path.dirname(checkpoint_path)
            print(f"Resuming from checkpoint file: {resume_checkpoint_path}")
        elif os.path.isdir(checkpoint_path):
            # It's a folder, find the latest checkpoint in it
            latest_checkpoint = find_latest_checkpoint(checkpoint_path)
            if latest_checkpoint:
                resume_checkpoint_path = latest_checkpoint
                output_dir = checkpoint_path
                print(f"Resuming from latest checkpoint in folder: {resume_checkpoint_path}")
            else:
                print(f"Error: No checkpoint files found in folder {checkpoint_path}")
                return None
        else:
            print(f"Error: Checkpoint path does not exist: {checkpoint_path}")
            return None
    
    if resume_checkpoint_path:
        # 3. Load previous config if resuming
        if args.resume_config_too:
            previous_config_path = os.path.join(output_dir, "training_config.yaml")
            
            if os.path.exists(previous_config_path):
                print(f"Loading previous config from: {previous_config_path}")
                config = load_config(previous_config_path)
                print("Successfully loaded previous config")
            else:
                print(f"Warning: Previous config file not found at {previous_config_path}")
                print("Using provided config file instead")
    
        # 4. Set the resume checkpoint path in config for train() function
        config.resume_checkpoint_path = resume_checkpoint_path
        config.output_dir = output_dir
    else:
        # 5. Handle output directory creation for new runs 
        # Get base model and dataset names for directory naming
        base_model_name = config.model_name.split('/')[-1]
        dataset_name = config.dataset_name
        
        # Create new instance - generate datetime and new directories
        current_time = datetime.datetime.now()
        datetime_str = current_time.strftime("%m%d%H%M")
        
        # Create new output directory path
        base_output_dir = args.output_dir_parent if args.output_dir_parent else config.base_output_dir
        config.output_dir = os.path.join(
            base_output_dir,
            f"{base_model_name}_{dataset_name}_{datetime_str}"
        )
        
        # Create directory
        os.makedirs(config.output_dir, exist_ok=True)
        print(f"Created new output directory: {config.output_dir}")
    
    print(f"Final config summary:")
    print(f"  Resume checkpoint: {resume_checkpoint_path}")
    print(f"  Output directory: {config.output_dir}")
    print(f"  Model: {config.model_name}")
    print(f"  Dataset: {config.dataset_name}")
    
    # Save configuration (after output directory is finalized and config is merged)
    save_config(config, config.output_dir)
    
    return config 