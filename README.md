# LLM Unlearning for SEMEval 2025

This project implements a machine unlearning approach for Large Language Models (LLMs) as part of the SEMEval 2025 shared task. The codebase has been refactored from a Jupyter notebook into a modular Python package following best practices.

## 🏗️ Project Structure

```
LLM_Unlearning_SEMEval2025/
├── config.py              # Configuration management
├── data_utils.py           # Data loading and preprocessing
├── model_utils.py          # Model loading and management
├── loss_functions.py       # Custom loss functions for unlearning
├── training_utils.py       # Training pipeline and utilities
├── task_vector.py          # Task vector operations
├── evaluation_utils.py     # Model evaluation utilities
├── main.py                 # Main execution script
├── setup.py                # Setup and installation script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── Copia_di_LLM_Unlearning.ipynb  # Original notebook (reference)
└── outputs/               # Generated outputs and results
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Install dependencies
python setup.py install

# Setup environment (create directories, etc.)
python setup.py setup

# Validate installation
python setup.py validate

# Run quick test
python setup.py test
```

### 2. Running the Pipeline

```bash
# Run the complete unlearning pipeline
python main.py
```

### 3. Configuration

Edit `config.py` to customize:

- Model paths and settings
- Training hyperparameters
- Loss function weights
- Evaluation parameters

## 📋 Key Components

### Configuration (`config.py`)

- Centralized configuration management using dataclasses
- Environment variable support for sensitive data (HF tokens)
- Easily configurable hyperparameters

### Data Processing (`data_utils.py`)

- Automated dataset downloading from HuggingFace Hub
- Data filtering and preprocessing
- Tokenization with start location tracking
- DataLoader creation with custom collation

### Model Management (`model_utils.py`)

- Model loading with 8-bit quantization
- LoRA (Low-Rank Adaptation) setup
- Memory management utilities
- Model saving and loading utilities

### Loss Functions (`loss_functions.py`)

- **Gradient Descent/Ascent Loss**: For forgetting/retaining specific content
- **Reverse KL Divergence**: To maintain general model capabilities
- **Random Disassociation Loss**: To prevent overfitting to specific patterns

### Training (`training_utils.py`)

- Gradient accumulation support
- Progress tracking with tqdm
- Automatic LoRA merging
- Checkpointing capabilities

### Task Vectors (`task_vector.py`)

- Task vector computation and operations
- Model arithmetic operations (addition, negation, scaling)
- Memory-efficient chunked application
- Task vector persistence and loading

### Evaluation (`evaluation_utils.py`)

- Perplexity and loss evaluation on retain/forget sets
- Text generation for qualitative assessment
- Comprehensive result logging and saving

## 🔧 Advanced Usage

### Custom Configuration

```python
from config import Config

# Create custom configuration
config = Config(
    learning_rate=5e-5,
    max_unlearn_steps=1000,
    bad_weight=2.0,
    debug_mode=True
)
```

### Modular Training

```python
from training_utils import UnlearningTrainer
from model_utils import ModelManager

# Initialize components
model_manager = ModelManager(config)
trainer = UnlearningTrainer(config)

# Load models
model, pretrained_model = model_manager.load_models()
model = model_manager.setup_lora(model)

# Train
unlearned_model = trainer.train_unlearning(
    model, pretrained_model, retain_loader, forget_loader,
    tokenizer, retain_answers
)
```

### Task Vector Operations

```python
from task_vector import TaskVector

# Create task vector
task_vector = TaskVector(pretrained_model, finetuned_model)

# Apply negated task vector for unlearning
neg_task_vector = -task_vector
unlearned_model = neg_task_vector.apply_to(pretrained_model, scaling_coef=2.0)
```

## 📊 Evaluation Metrics

The pipeline evaluates models using:

- **Negative Log-Likelihood (NLL)** on retain and forget sets
- **Perplexity (PPL)** measurements
- **Qualitative text generation** samples
- **Before/after comparisons**

## 🎯 SEMEval 2025 Tasks

The project supports all three SEMEval 2025 unlearning subtasks:

- **Task 1**: Long-form synthetic creative documents
- **Task 2**: Short-form synthetic biographies with PII
- **Task 3**: Real documents from training data

Configure the target task by setting `target_task` in `config.py`.

## 💾 Memory Optimization

The codebase includes several memory optimization techniques:

- 8-bit model quantization using BitsAndBytes
- LoRA for parameter-efficient fine-tuning
- Gradient checkpointing
- Memory cleanup utilities
- Chunked task vector application

## 🔍 Debugging and Logging

- Comprehensive logging throughout the pipeline
- Debug mode for smaller dataset testing
- Gradient monitoring during training
- Error handling and recovery

## 📁 Output Files

The pipeline generates:

- `evaluation_results.json`: Quantitative evaluation metrics
- Model checkpoints in `semeval25-unlearning-model/`
- Task vector in `semeval25-unlearning-model/task_vector/`
- Training logs and samples

## 🤝 Contributing

When adding new features:

1. Follow the existing modular structure
2. Add appropriate logging
3. Update configuration if needed
4. Include error handling
5. Update documentation

## 📄 License

This project is part of the SEMEval 2025 shared task. Please refer to the competition guidelines for usage restrictions.

## 🔗 References

- SEMEval 2025 Task: Machine Unlearning for Language Models
- Original research on machine unlearning techniques
- HuggingFace Transformers and PEFT libraries
- Task Vector methodology for model editingLLM_Unlearning_SEMEval2025
  Submission for SEMEval 2025 Unlearning Challenge. We apply the method from “Towards Safer Large Language Models through Machine Unlearning” by Liu et al. to selectively forget data in LLMs. Includes baseline integration, official metrics, and support for custom methods.
