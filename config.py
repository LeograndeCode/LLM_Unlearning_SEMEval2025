"""Configuration file for LLM Unlearning project."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration class for the LLM unlearning project."""

    # Model and dataset settings
    model_name: str = "allenai/OLMo-1B-0724-hf"
    pretrained_model_path: str = "semeval25-unlearning-1B-model"
    dataset_repo: str = (
        "llmunlearningsemeval2025organization/semeval25-unlearning-dataset-public"
    )
    model_repo: str = (
        "llmunlearningsemeval2025organization/olmo-1B-model-semeval25-unlearning"
    )

    # HuggingFace token (should be set via environment variable)
    hf_token: str = os.getenv("HF_TOKEN", "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")

    # Training hyperparameters
    batch_size: int = 1
    learning_rate: float = 1e-4
    max_unlearn_steps: int = 2000
    accumulation_steps: int = 4
    max_length: int = 128

    # Loss weights
    bad_weight: float = 1.0
    random_weight: float = 1.0
    normal_weight: float = 0.5

    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_target_modules: list = None

    # Task vector settings
    task_vector_scaling_coef: float = 2.0

    # Evaluation settings
    eval_batch_size: int = 1
    max_new_tokens: int = 50
    num_eval_samples: int = 5

    # Paths
    model_save_dir: str = "semeval25-unlearning-model"
    task_vector_saving_path: str = "semeval25-unlearning-model/task_vector"
    temp_model_dir: str = "tmp/unlearned_8bit"

    # Data filtering
    target_task: str = "Task2"

    # Debug settings
    debug_mode: bool = False
    debug_sample_size: int = 100

    def __post_init__(self):
        """Initialize default values that depend on other attributes."""
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "v_proj"]

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create Config instance from dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert Config instance to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
