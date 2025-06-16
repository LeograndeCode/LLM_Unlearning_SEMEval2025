"""Training utilities for LLM Unlearning project."""

import logging
import torch
from accelerate import Accelerator
from transformers import get_scheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from typing import Tuple, List
from tqdm.auto import tqdm

from config import Config
from loss_functions import get_answer_loss, compute_reverse_kl, get_rand_ans_loss

logger = logging.getLogger(__name__)


class UnlearningTrainer:
    """Handles the unlearning training process."""

    def __init__(self, config: Config):
        self.config = config
        self.accelerator = Accelerator()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup_optimizer_and_scheduler(
        self, model: torch.nn.Module
    ) -> Tuple[AdamW, any]:
        """Setup optimizer and learning rate scheduler."""
        optimizer = AdamW(model.parameters(), lr=self.config.learning_rate)

        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=self.config.max_unlearn_steps,
        )

        return optimizer, lr_scheduler

    def train_unlearning(
        self,
        model: torch.nn.Module,
        pretrained_model: torch.nn.Module,
        retain_loader: DataLoader,
        forget_loader: DataLoader,
        tokenizer: any,
        retain_answers: List[str],
    ) -> torch.nn.Module:
        """
        Main unlearning training loop.

        Args:
            model: Model to be unlearned (with LoRA)
            pretrained_model: Reference pretrained model
            retain_loader: DataLoader for retain data
            forget_loader: DataLoader for forget data
            tokenizer: Tokenizer instance
            retain_answers: List of retain answers for random disassociation

        Returns:
            Trained model
        """
        logger.info("Starting unlearning training...")

        # Enable anomaly detection for debugging
        torch.autograd.set_detect_anomaly(True)

        # Setup optimizer and scheduler
        optimizer, lr_scheduler = self.setup_optimizer_and_scheduler(model)

        # Training loop
        optimizer.zero_grad()
        idx = 0
        step = 0

        progress_bar = tqdm(total=self.config.max_unlearn_steps, desc="Unlearning")

        while idx < self.config.max_unlearn_steps:
            for bad_batch, normal_batch in zip(forget_loader, retain_loader):
                # Compute all losses
                bad_loss = get_answer_loss("gd", bad_batch, model, self.device)
                random_loss = get_rand_ans_loss(
                    bad_batch, tokenizer, retain_answers, model, device=self.device
                )
                normal_loss = compute_reverse_kl(
                    pretrained_model, model, normal_batch, self.device
                )

                # Combined loss
                loss = (
                    self.config.bad_weight * bad_loss
                    + self.config.random_weight * random_loss
                    + self.config.normal_weight * normal_loss
                ) / self.config.accumulation_steps

                # Log individual losses
                if idx % 100 == 0:
                    logger.info(f"Step {idx}:")
                    logger.info(f"  GD loss: {bad_loss.item():.4f}")
                    logger.info(f"  RD loss: {random_loss.item():.4f}")
                    logger.info(f"  revKL loss: {normal_loss.item():.4f}")
                    logger.info(
                        f"  Combined loss: {(loss * self.config.accumulation_steps).item():.4f}"
                    )

                # Backward pass
                self.accelerator.backward(loss)

                # Log gradients for debugging
                if idx % 100 == 0:
                    for n, p in model.named_parameters():
                        if "lora" in n and p.grad is not None:
                            logger.debug(
                                f"{n} grad mean: {p.grad.abs().mean().item():.6f}"
                            )

                # Optimizer step every accumulation_steps
                if (step + 1) % self.config.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                idx += 1
                step += 1
                progress_bar.update(1)

                if idx >= self.config.max_unlearn_steps:
                    break

        progress_bar.close()

        # Merge and unload LoRA weights
        if hasattr(model, "merge_and_unload"):
            logger.info("Merging and unloading LoRA weights...")
            model = model.merge_and_unload()

        logger.info("Unlearning training completed!")
        return model

    def save_training_checkpoint(self, model: torch.nn.Module, step: int) -> None:
        """Save training checkpoint."""
        checkpoint_path = f"{self.config.model_save_dir}/checkpoint-{step}"
        model.save_pretrained(checkpoint_path, from_pt=True)
        logger.info(f"Checkpoint saved at step {step}")

    def load_training_checkpoint(self, checkpoint_path: str) -> torch.nn.Module:
        """Load training checkpoint."""
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path, torch_dtype=torch.float32, device_map="auto"
        )
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return model
