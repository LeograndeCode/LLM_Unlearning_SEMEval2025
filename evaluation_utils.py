"""Evaluation utilities for LLM Unlearning project."""

import logging
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from typing import Tuple, List, Dict, Any
import pandas as pd

from config import Config

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Handles model evaluation for unlearning experiments."""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def eval_loss(
        self, model: torch.nn.Module, dataloader: DataLoader, tokenizer: AutoTokenizer
    ) -> Tuple[float, float]:
        """
        Evaluate model loss and perplexity on a dataset.

        Args:
            model: Model to evaluate
            dataloader: DataLoader for evaluation data
            tokenizer: Tokenizer instance

        Returns:
            Tuple of (average_nll, perplexity)
        """
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        loss_fct = torch.nn.CrossEntropyLoss(
            ignore_index=tokenizer.pad_token_id, reduction="sum"
        )

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = model(input_ids, attention_mask=attention_mask)
                # logits: [B, L, V]
                shift_logits = outputs.logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                # flatten
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )
                total_loss += loss.item()
                total_tokens += (shift_labels != tokenizer.pad_token_id).sum().item()

        avg_nll = total_loss / total_tokens
        ppl = torch.exp(torch.tensor(avg_nll))
        return avg_nll, ppl.item()

    def evaluate_models(
        self,
        pretrained_model: torch.nn.Module,
        unlearned_model: torch.nn.Module,
        retain_loader: DataLoader,
        forget_loader: DataLoader,
        tokenizer: AutoTokenizer,
    ) -> Dict[str, float]:
        """
        Evaluate both pretrained and unlearned models on retain and forget sets.

        Args:
            pretrained_model: Original pretrained model
            unlearned_model: Model after unlearning
            retain_loader: DataLoader for retain validation data
            forget_loader: DataLoader for forget validation data
            tokenizer: Tokenizer instance

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating models...")

        # Evaluate pretrained model
        logger.info("Evaluating pretrained model on forget set...")
        nll_forget_pre, ppl_forget_pre = self.eval_loss(
            pretrained_model, forget_loader, tokenizer
        )

        logger.info("Evaluating pretrained model on retain set...")
        nll_retain_pre, ppl_retain_pre = self.eval_loss(
            pretrained_model, retain_loader, tokenizer
        )

        # Evaluate unlearned model
        logger.info("Evaluating unlearned model on forget set...")
        nll_forget_post, ppl_forget_post = self.eval_loss(
            unlearned_model, forget_loader, tokenizer
        )

        logger.info("Evaluating unlearned model on retain set...")
        nll_retain_post, ppl_retain_post = self.eval_loss(
            unlearned_model, retain_loader, tokenizer
        )

        results = {
            "nll_forget_pre": nll_forget_pre,
            "ppl_forget_pre": ppl_forget_pre,
            "nll_forget_post": nll_forget_post,
            "ppl_forget_post": ppl_forget_post,
            "nll_retain_pre": nll_retain_pre,
            "ppl_retain_pre": ppl_retain_pre,
            "nll_retain_post": nll_retain_post,
            "ppl_retain_post": ppl_retain_post,
        }

        # Log results
        logger.info("Evaluation Results:")
        logger.info(
            f"Forget Set - Pretrained: NLL={nll_forget_pre:.2f}, PPL={ppl_forget_pre:.2f}"
        )
        logger.info(
            f"Forget Set - Unlearned:  NLL={nll_forget_post:.2f}, PPL={ppl_forget_post:.2f}"
        )
        logger.info(
            f"Retain Set - Pretrained: NLL={nll_retain_pre:.2f}, PPL={ppl_retain_pre:.2f}"
        )
        logger.info(
            f"Retain Set - Unlearned:  NLL={nll_retain_post:.2f}, PPL={ppl_retain_post:.2f}"
        )

        return results

    def generate_samples(
        self,
        pretrained_model: torch.nn.Module,
        unlearned_model: torch.nn.Module,
        forget_validation_df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        num_samples: int = None,
    ) -> List[Dict[str, str]]:
        """
        Generate text samples to compare pretrained and unlearned models.

        Args:
            pretrained_model: Original pretrained model
            unlearned_model: Model after unlearning
            forget_validation_df: DataFrame with forget validation data
            tokenizer: Tokenizer instance
            num_samples: Number of samples to generate (default from config)

        Returns:
            List of dictionaries with prompt, original output, and unlearned output
        """
        if num_samples is None:
            num_samples = self.config.num_eval_samples

        logger.info(f"Generating {num_samples} text samples...")

        samples = []
        sample_df = forget_validation_df.sample(num_samples)

        pretrained_model.eval()
        unlearned_model.eval()

        with torch.no_grad():
            for index, example in sample_df.iterrows():
                prompt = example["input"]

                # Tokenize prompt
                prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(
                    self.device
                )

                # Generate with pretrained model
                out_pre = pretrained_model.generate(
                    prompt_tokens,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id,
                )

                # Generate with unlearned model
                out_post = unlearned_model.generate(
                    prompt_tokens,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id,
                )

                # Decode outputs
                orig_text = tokenizer.decode(out_pre[0], skip_special_tokens=True)
                unlearned_text = tokenizer.decode(out_post[0], skip_special_tokens=True)

                samples.append(
                    {
                        "prompt": prompt,
                        "original": orig_text,
                        "unlearned": unlearned_text,
                        "ground_truth": example["output"],
                    }
                )

        return samples

    def print_generation_samples(self, samples: List[Dict[str, str]]) -> None:
        """Print generation samples in a readable format."""
        logger.info("Generation Samples:")
        logger.info("=" * 80)

        for i, sample in enumerate(samples, 1):
            logger.info(f"Sample {i}:")
            logger.info(f"PROMPT: {sample['prompt']}")
            logger.info(f"ORIGINAL: {sample['original']}")
            logger.info(f"UNLEARNED: {sample['unlearned']}")
            logger.info("-" * 40)

    def save_evaluation_results(
        self,
        results: Dict[str, float],
        samples: List[Dict[str, str]],
        output_path: str = "evaluation_results.json",
    ) -> None:
        """Save evaluation results to file."""
        import json

        output_data = {
            "metrics": results,
            "samples": samples,
            "config": self.config.to_dict(),
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Evaluation results saved to {output_path}")
