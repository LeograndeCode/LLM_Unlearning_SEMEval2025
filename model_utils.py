"""Model setup and utilities for LLM Unlearning project."""

import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig
from typing import Tuple

from config import Config

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model loading, configuration, and setup."""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def setup_quantization_config(self) -> BitsAndBytesConfig:
        """Setup 8-bit quantization configuration."""
        return BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)

    def load_models(self) -> Tuple[torch.nn.Module, torch.nn.Module]:
        """Load pretrained and quantized models."""
        logger.info("Loading models...")

        # Setup quantization
        bnb_config = self.setup_quantization_config()

        # Load quantized model for training
        model = AutoModelForCausalLM.from_pretrained(
            self.config.pretrained_model_path,
            quantization_config=bnb_config,
            device_map="auto",
        )

        # Remove clip_qkv if present
        if hasattr(model.config, "clip_qkv"):
            model.config.clip_qkv = None

        # Load full precision model for reference
        pretrained_model = AutoModelForCausalLM.from_pretrained(
            self.config.pretrained_model_path, device_map="auto"
        )

        return model, pretrained_model

    def setup_lora(self, model: torch.nn.Module) -> torch.nn.Module:
        """Setup LoRA configuration for the model."""
        logger.info("Setting up LoRA...")

        # Enable gradient checkpointing
        model.gradient_checkpointing_enable()

        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)

        # Create LoRA configuration
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            inference_mode=False,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        return model

    def load_tokenizer(self) -> AutoTokenizer:
        """Load and return tokenizer."""
        return AutoTokenizer.from_pretrained(self.config.model_name)

    def save_model(
        self, model: torch.nn.Module, path: str, merge_lora: bool = False
    ) -> None:
        """Save model to specified path."""
        logger.info(f"Saving model to {path}")

        if merge_lora and hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()

        model.save_pretrained(path, from_pt=True)

    def load_model_from_path(
        self, path: str, torch_dtype: torch.dtype = torch.float32
    ) -> torch.nn.Module:
        """Load model from specified path."""
        logger.info(f"Loading model from {path}")

        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch_dtype, device_map="auto"
        )

        return model

    def cleanup_memory(self) -> None:
        """Clean up GPU memory."""
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU memory cleaned up")
