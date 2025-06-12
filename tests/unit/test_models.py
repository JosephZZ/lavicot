"""Basic tests for LaViCoT models."""

import pytest
import torch
import sys
import os

# Path is handled by pytest configuration

from src.lavicot.models.lavicot_bias import TestTimePrefixModel, create_test_time_prefix_config


class TestLaViCotModels:
    """Test LaViCoT model functionality."""
    
    def test_prefix_config_creation(self):
        """Test creating a prefix configuration."""
        config = create_test_time_prefix_config(
            layer_selection_mode="spread",
            layer_selection=3,
            max_iterations=5
        )
        
        assert config.layer_selection_mode == "spread"
        assert config.layer_selection == 3
        assert config.max_iterations == 5
    
    def test_prefix_config_defaults(self):
        """Test default configuration creation."""
        config = create_test_time_prefix_config()
        
        assert config.layer_selection_mode == "all"
        assert config.max_iterations == 10
        assert config.gradient_steps == 4
    
    @pytest.mark.slow
    def test_model_loading(self):
        """Test loading a small model with prefix generation."""
        # This test requires GPU and is marked as slow
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from lavicot.models.lavicot_bias import add_instance_level_prefix_generator
        
        model_name = "distilgpt2"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add prefix generation capability
        prefix_model = add_instance_level_prefix_generator(model)
        
        assert hasattr(prefix_model, 'prefix_generators')
        assert hasattr(prefix_model, 'reset_prefixes')
        assert hasattr(prefix_model, 'update_prefix_given_input')
        
        # Clean up
        prefix_model.cleanup() 