#!/usr/bin/env python3
"""Utility script for common operations in LLM Unlearning project."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )
    return logging.getLogger(__name__)


def run_training_only():
    """Run only the training phase."""
    from config import Config
    from data_utils import DataProcessor
    from model_utils import ModelManager
    from training_utils import UnlearningTrainer

    logger = setup_logging()
    logger.info("Running training-only pipeline...")

    config = Config()
    data_processor = DataProcessor(config)
    model_manager = ModelManager(config)
    trainer = UnlearningTrainer(config)

    # Load data
    retain_train_df, _, forget_train_df, _ = data_processor.load_datasets()
    ds_retain, ds_forget = data_processor.create_datasets(
        retain_train_df, forget_train_df
    )
    retain_loader, forget_loader = data_processor.create_dataloaders(
        ds_retain, ds_forget
    )

    # Load models
    model, pretrained_model = model_manager.load_models()
    model = model_manager.setup_lora(model)
    tokenizer = model_manager.load_tokenizer()

    # Train
    retain_answers = retain_train_df["output"].tolist()
    unlearned_model = trainer.train_unlearning(
        model, pretrained_model, retain_loader, forget_loader, tokenizer, retain_answers
    )

    # Save
    model_manager.save_model(unlearned_model, config.temp_model_dir)
    logger.info("Training completed!")


def run_evaluation_only():
    """Run only the evaluation phase."""
    from config import Config
    from data_utils import DataProcessor
    from model_utils import ModelManager
    from evaluation_utils import ModelEvaluator

    logger = setup_logging()
    logger.info("Running evaluation-only pipeline...")

    config = Config()
    data_processor = DataProcessor(config)
    model_manager = ModelManager(config)
    evaluator = ModelEvaluator(config)

    # Load validation data
    _, retain_validation_df, _, forget_validation_df = data_processor.load_datasets()
    ds_retain_val, ds_forget_val = data_processor.create_datasets(
        retain_validation_df, forget_validation_df
    )
    retain_val_loader, forget_val_loader = data_processor.create_dataloaders(
        ds_retain_val, ds_forget_val
    )

    # Load models
    tokenizer = model_manager.load_tokenizer()
    pretrained_model = model_manager.load_model_from_path(config.pretrained_model_path)
    unlearned_model = model_manager.load_model_from_path(config.task_vector_saving_path)

    # Evaluate
    results = evaluator.evaluate_models(
        pretrained_model,
        unlearned_model,
        retain_val_loader,
        forget_val_loader,
        tokenizer,
    )

    # Generate samples
    samples = evaluator.generate_samples(
        pretrained_model, unlearned_model, forget_validation_df, tokenizer
    )

    evaluator.print_generation_samples(samples)
    evaluator.save_evaluation_results(results, samples)
    logger.info("Evaluation completed!")


def create_task_vector():
    """Create and apply task vector."""
    from config import Config
    from model_utils import ModelManager
    from task_vector import TaskVector

    logger = setup_logging()
    logger.info("Creating and applying task vector...")

    config = Config()
    model_manager = ModelManager(config)

    # Load models
    pretrained_model = model_manager.load_model_from_path(config.pretrained_model_path)
    unlearned_model = model_manager.load_model_from_path(config.temp_model_dir)

    # Create task vector
    task_vector = TaskVector(pretrained_model, unlearned_model)
    neg_task_vector = -task_vector

    # Apply task vector
    final_model = neg_task_vector.apply_to(
        pretrained_model, scaling_coef=config.task_vector_scaling_coef
    )

    # Save
    model_manager.save_model(final_model, config.task_vector_saving_path)
    logger.info("Task vector applied and model saved!")


def debug_data():
    """Debug data loading and preprocessing."""
    from config import Config
    from data_utils import DataProcessor

    logger = setup_logging()
    logger.info("Debugging data loading...")

    config = Config(debug_mode=True)  # Enable debug mode
    data_processor = DataProcessor(config)

    # Load and inspect data
    retain_train_df, retain_val_df, forget_train_df, forget_val_df = (
        data_processor.load_datasets()
    )

    logger.info(f"Retain train: {len(retain_train_df)} examples")
    logger.info(f"Retain val: {len(retain_val_df)} examples")
    logger.info(f"Forget train: {len(forget_train_df)} examples")
    logger.info(f"Forget val: {len(forget_val_df)} examples")

    # Show sample data
    logger.info("Sample retain example:")
    logger.info(f"Input: {retain_train_df.iloc[0]['input'][:100]}...")
    logger.info(f"Output: {retain_train_df.iloc[0]['output'][:100]}...")

    logger.info("Sample forget example:")
    logger.info(f"Input: {forget_train_df.iloc[0]['input'][:100]}...")
    logger.info(f"Output: {forget_train_df.iloc[0]['output'][:100]}...")


def clean_outputs():
    """Clean output directories."""
    import shutil

    logger = setup_logging()
    logger.info("Cleaning output directories...")

    directories_to_clean = [
        "semeval25-unlearning-model",
        "tmp",
        "train",
        "validation",
        "logs",
        "outputs",
    ]

    for directory in directories_to_clean:
        path = Path(directory)
        if path.exists():
            shutil.rmtree(path)
            logger.info(f"Removed directory: {directory}")
        else:
            logger.info(f"Directory does not exist: {directory}")


def main():
    """Main utility function."""
    parser = argparse.ArgumentParser(
        description="Utility script for LLM Unlearning project"
    )
    parser.add_argument(
        "action",
        choices=["train", "eval", "task-vector", "debug-data", "clean"],
        help="Action to perform",
    )

    args = parser.parse_args()

    try:
        if args.action == "train":
            run_training_only()
        elif args.action == "eval":
            run_evaluation_only()
        elif args.action == "task-vector":
            create_task_vector()
        elif args.action == "debug-data":
            debug_data()
        elif args.action == "clean":
            clean_outputs()

        print(f"Action '{args.action}' completed successfully!")

    except Exception as e:
        print(f"Action '{args.action}' failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
