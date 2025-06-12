"""Configuration loader for LaViCoT training."""

import os
import yaml
from types import SimpleNamespace
from typing import Dict, Any, Optional

def load_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """Load dataset configuration from the datasets directory.
    
    Args:
        dataset_name: Name of the dataset config file (without .yaml extension)
        
    Returns:
        Dictionary with dataset configuration
    """
    dataset_config_path = os.path.join(
        os.path.dirname(__file__), 
        "datasets", 
        f"{dataset_name}.yaml"
    )
    
    if not os.path.exists(dataset_config_path):
        raise FileNotFoundError(f"Dataset config file not found: {dataset_config_path}")
    
    with open(dataset_config_path, 'r') as f:
        return yaml.safe_load(f)

def load_config(config_path: Optional[str] = None, dataset_config_name: Optional[str] = None, **kwargs) -> SimpleNamespace:
    """Load configuration from YAML file and override with command-line arguments.
    
    Args:
        config_path: Path to the configuration file. If None, uses default config.
        dataset_config_name: Name of the dataset config to load (e.g., 'gsm8k', 'math')
        **kwargs: Command-line arguments to override config values.
        
    Returns:
        SimpleNamespace object with merged configuration.
    """
    # Start with default config path if none provided
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "defaults", "default.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    # Load main YAML file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load and merge dataset configuration if specified
    if dataset_config_name:
        dataset_config = load_dataset_config(dataset_config_name)
        config.update(dataset_config)
    elif 'dataset_config_name' in config:
        # Load dataset config from main config if specified there
        dataset_config = load_dataset_config(config['dataset_config_name'])
        config.update(dataset_config)
        # Remove the dataset_config_name key as it's no longer needed
        del config['dataset_config_name']
    
    # Override with command-line arguments
    config.update(kwargs)
    
    # Convert numeric strings to appropriate types
    for key, value in config.items():
        if isinstance(value, str):
            # Try to convert to float first
            try:
                config[key] = float(value)
                # If it's an integer, convert to int
                if config[key].is_integer():
                    config[key] = int(config[key])
            except ValueError:
                pass
    
    # Convert to SimpleNamespace for dot notation access
    return SimpleNamespace(**config)

def save_config(config: SimpleNamespace, output_dir: str) -> None:
    """Save configuration to a YAML file.
    
    Args:
        config: SimpleNamespace object to save
        output_dir: Directory to save the config file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert SimpleNamespace to dict
    config_dict = vars(config)
    
    # Convert to YAML
    yaml_str = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)
    
    # Save to file
    with open(os.path.join(output_dir, "training_config.yaml"), "w") as f:
        f.write(yaml_str) 