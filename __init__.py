"""LLM Unlearning for SEMEval 2025 Package."""

__version__ = "1.0.0"
__author__ = "SEMEval 2025 Team"
__description__ = "Machine unlearning implementation for Large Language Models"

# Import main components for easier access
from .config import Config
from .data_utils import DataProcessor
from .model_utils import ModelManager
from .training_utils import UnlearningTrainer
from .evaluation_utils import ModelEvaluator
from .task_vector import TaskVector

__all__ = [
    "Config",
    "DataProcessor",
    "ModelManager",
    "UnlearningTrainer",
    "ModelEvaluator",
    "TaskVector",
]
