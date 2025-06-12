# Dataset Configurations

This directory contains dataset-specific configuration files that can be used with LaViCoT training and evaluation.

## Available Datasets

- **gsm8k.yaml**: Grade School Math 8K dataset configuration
- **math.yaml**: MATH competition dataset configuration  
- **aqr.yaml**: Algebraic Reasoning dataset configuration

## Usage

### Training with a specific dataset

```bash
# Use GSM8K dataset
python scripts/train.py --config ./src/lavicot/config/defaults/default.yaml --dataset gsm8k

# Use MATH dataset
python scripts/train.py --config ./src/lavicot/config/defaults/default.yaml --dataset math
```

### Evaluation with a specific dataset

```bash
# Evaluate on GSM8K
python scripts/evaluate.py --config ./src/lavicot/config/defaults/default.yaml --dataset gsm8k --model_name path/to/model

# Evaluate on MATH dataset
python scripts/evaluate.py --config ./src/lavicot/config/defaults/default.yaml --dataset math --model_name path/to/model
```

### Using dataset config in main config file

You can also specify the dataset config directly in your main configuration file:

```yaml
# In your main config file (e.g., default.yaml)
dataset_config_name: "gsm8k"  # This will load gsm8k.yaml
```

## Creating New Dataset Configurations

To add a new dataset, create a new YAML file in this directory with the following structure:

```yaml
# Example: my_dataset.yaml
dataset_name: "my_dataset"
dataset_config: "main"  # or null if no sub-config needed
num_eval_samples: "full"  # or specific number
num_train_samples: 5000  # or "full"

# Train/test split configuration (for datasets that only have train split)
train_val_split_ratio: 0.8  # Split train data into train(80%)/test(20%)
# Set to null if the dataset has existing train/test splits

# Optional metadata
description: "Description of your dataset"
task_type: "reasoning"  # or other task type
difficulty: "medium"  # optional difficulty level
```

## Configuration Priority

The dataset configuration loading follows this priority:

1. Command-line `--dataset` parameter (highest priority)
2. `dataset_config_name` specified in the main config file
3. Default dataset config embedded in main config (if any)

When a dataset config is loaded, it will override any dataset-related parameters in the main configuration file.

## Train/Test Split Handling

Some datasets only provide a training split. For these cases, you can use the `train_val_split_ratio` parameter:

- **`train_val_split_ratio: null`** - Use existing train/test splits from the dataset
- **`train_val_split_ratio: 0.8`** - Split the train data into 80% train / 20% test
- **`train_val_split_ratio: 0.85`** - Split the train data into 85% train / 15% test

### Examples:

**Dataset with existing train/test splits (like GSM8K):**
```yaml
dataset_name: "gsm8k"
train_val_split_ratio: null  # Use existing splits
```

**Dataset with only train split (like some MATH datasets):**
```yaml
dataset_name: "my_dataset_with_only_train"
train_val_split_ratio: 0.8  # Create 80/20 split from train data
```

The splitting is done deterministically using the `seed` parameter from your main configuration file. 