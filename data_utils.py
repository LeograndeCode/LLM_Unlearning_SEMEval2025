"""Data loading and preprocessing utilities for LLM Unlearning project."""

import logging
import pandas as pd
import torch
from datasets import Dataset
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from typing import Dict, Any, List, Tuple

from config import Config

logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles data loading, preprocessing, and tokenization."""

    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    def download_data(self) -> None:
        """Download model and dataset from HuggingFace Hub."""
        logger.info("Downloading model...")
        snapshot_download(
            repo_id=self.config.model_repo,
            token=self.config.hf_token,
            local_dir=self.config.pretrained_model_path,
        )

        logger.info("Downloading dataset...")
        snapshot_download(
            repo_id=self.config.dataset_repo,
            token=self.config.hf_token,
            local_dir="semeval25-unlearning-data",
            repo_type="dataset",
        )

    def load_datasets(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and return retain/forget train/validation datasets."""
        logger.info("Loading datasets...")

        retain_train_df = pd.read_parquet(
            "semeval25-unlearning-data/data/retain_train-00000-of-00001.parquet",
            engine="pyarrow",
        )
        retain_validation_df = pd.read_parquet(
            "semeval25-unlearning-data/data/retain_validation-00000-of-00001.parquet",
            engine="pyarrow",
        )
        forget_train_df = pd.read_parquet(
            "semeval25-unlearning-data/data/forget_train-00000-of-00001.parquet",
            engine="pyarrow",
        )
        forget_validation_df = pd.read_parquet(
            "semeval25-unlearning-data/data/forget_validation-00000-of-00001.parquet",
            engine="pyarrow",
        )

        # Filter by target task
        if self.config.target_task:
            logger.info(f"Filtering data for task: {self.config.target_task}")
            retain_train_df = retain_train_df[
                retain_train_df["task"] == self.config.target_task
            ]
            retain_validation_df = retain_validation_df[
                retain_validation_df["task"] == self.config.target_task
            ]
            forget_train_df = forget_train_df[
                forget_train_df["task"] == self.config.target_task
            ]
            forget_validation_df = forget_validation_df[
                forget_validation_df["task"] == self.config.target_task
            ]

        # Debug mode: sample smaller datasets
        if self.config.debug_mode:
            logger.info(
                f"Debug mode: sampling {self.config.debug_sample_size} examples"
            )
            sample_size = self.config.debug_sample_size
            retain_train_df = retain_train_df.sample(
                n=sample_size, random_state=42
            ).reset_index(drop=True)
            forget_train_df = forget_train_df.sample(
                n=sample_size, random_state=42
            ).reset_index(drop=True)
            retain_validation_df = retain_validation_df.sample(
                n=sample_size // 10, random_state=42
            ).reset_index(drop=True)
            forget_validation_df = forget_validation_df.sample(
                n=sample_size // 10, random_state=42
            ).reset_index(drop=True)

        return (
            retain_train_df,
            retain_validation_df,
            forget_train_df,
            forget_validation_df,
        )

    def tokenize_with_start(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize example with start location tracking."""
        q, a = example["input"], example["output"]
        prefix = q
        full = q + a

        # Tokenize prefix to count real tokens (no padding)
        t_pref = self.tokenizer(prefix, truncation=True, padding=False)
        start_locs = len(t_pref["input_ids"])

        # Tokenize full text with padding/truncation
        t_full = self.tokenizer(
            full,
            truncation=True,
            padding="max_length",
            max_length=self.config.max_length,
        )

        return {
            "input_ids": t_full["input_ids"],
            "attention_mask": t_full["attention_mask"],
            "labels": t_full["input_ids"],
            "start_locs": start_locs,
        }

    def create_datasets(
        self, retain_df: pd.DataFrame, forget_df: pd.DataFrame
    ) -> Tuple[Dataset, Dataset]:
        """Create HuggingFace datasets from dataframes."""
        logger.info("Creating and tokenizing datasets...")

        ds_retain = Dataset.from_pandas(retain_df).map(
            self.tokenize_with_start, batched=False, load_from_cache_file=False
        )

        ds_forget = Dataset.from_pandas(forget_df).map(
            self.tokenize_with_start, batched=False, load_from_cache_file=False
        )

        return ds_retain, ds_forget

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate function for DataLoader."""
        return {
            "input_ids": torch.tensor([x["input_ids"] for x in batch]),
            "attention_mask": torch.tensor([x["attention_mask"] for x in batch]),
            "labels": torch.tensor([x["labels"] for x in batch]),
            "start_locs": torch.tensor([x["start_locs"] for x in batch]),
        }

    def create_dataloaders(
        self, ds_retain: Dataset, ds_forget: Dataset
    ) -> Tuple[DataLoader, DataLoader]:
        """Create DataLoaders for training."""
        logger.info("Creating DataLoaders...")

        retain_loader = DataLoader(
            ds_retain,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

        forget_loader = DataLoader(
            ds_forget,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

        return retain_loader, forget_loader

    def save_jsonl_files(
        self,
        retain_train_df: pd.DataFrame,
        forget_train_df: pd.DataFrame,
        retain_validation_df: pd.DataFrame,
        forget_validation_df: pd.DataFrame,
    ) -> None:
        """Save datasets as JSONL files."""
        import os

        os.makedirs("train", exist_ok=True)
        os.makedirs("validation", exist_ok=True)

        retain_train_df.to_json("train/retain.jsonl", orient="records", lines=True)
        forget_train_df.to_json("train/forget.jsonl", orient="records", lines=True)
        retain_validation_df.to_json(
            "validation/retain.jsonl", orient="records", lines=True
        )
        forget_validation_df.to_json(
            "validation/forget.jsonl", orient="records", lines=True
        )

        logger.info("Saved datasets as JSONL files")
