"""Configuration loader for LaViCoT training."""

import os
import yaml
from types import SimpleNamespace
from typing import Dict, Any, Optional

def load_config(config_path: Optional[str] = None, **kwargs) -> SimpleNamespace:
    """Load configuration from YAML file and override with command-line arguments.
    
    Args:
        config_path: Path to the configuration file. If None, uses default config.
        **kwargs: Command-line arguments to override config values.
        
    Returns:
        SimpleNamespace object with merged configuration.
    """
    # Start with default config path if none provided
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "default_config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    # Load YAML file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
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