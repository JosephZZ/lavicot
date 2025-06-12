"""LaViCoT: Test-Time Prefix Generation for Enhanced Reasoning.

This package provides tools for training and evaluating language models
with test-time prefix generation capabilities.
"""

__version__ = "0.1.0"

# Core model exports - using the bias model as default
from .models.lavicot_bias import (
    TestTimePrefixModel,
    TestTimePrefixConfig,
    create_test_time_prefix_config,
    add_instance_level_prefix_generator
)

# Training exports
from .training.trainer import LaviCotTrainer
from .config.config_loader import load_config, save_config

__all__ = [
    "__version__",
    "TestTimePrefixModel",
    "TestTimePrefixConfig", 
    "create_test_time_prefix_config",
    "add_instance_level_prefix_generator",
    "LaviCotTrainer",
    "load_config",
    "save_config",
]
