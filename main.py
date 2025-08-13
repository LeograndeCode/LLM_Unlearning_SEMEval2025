"""Main execution script for LLM Unlearning project."""

import logging
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import Config
from data_utils import DataProcessor
from model_utils import ModelManager
from training_utils import UnlearningTrainer
from evaluation_utils import ModelEvaluator
from task_vector import TaskVector


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )
    return logging.getLogger(__name__)


def main():
    """Main execution function."""
    logger = setup_logging()
    logger.info("Starting LLM Unlearning pipeline...")

    # Load configuration
    config = Config()
    logger.info(f"Configuration loaded. Target task: {config.target_task}")

    # Initialize components
    data_processor = DataProcessor(config)
    model_manager = ModelManager(config)
    trainer = UnlearningTrainer(config)
    evaluator = ModelEvaluator(config)

    try:
        # Step 1: Download and load data
        logger.info("=" * 50)
        logger.info("STEP 1: Data Loading and Preprocessing")
        logger.info("=" * 50)

        data_processor.download_data()
        retain_train_df, retain_validation_df, forget_train_df, forget_validation_df = (
            data_processor.load_datasets()
        )

        # Save JSONL files for reference
        data_processor.save_jsonl_files(
            retain_train_df, forget_train_df, retain_validation_df, forget_validation_df
        )

        # Create datasets and dataloaders
        ds_retain_train, ds_forget_train = data_processor.create_datasets(
            retain_train_df, forget_train_df
        )
        ds_retain_val, ds_forget_val = data_processor.create_datasets(
            retain_validation_df, forget_validation_df
        )

        retain_train_loader, forget_train_loader = data_processor.create_dataloaders(
            ds_retain_train, ds_forget_train
        )
        retain_val_loader, forget_val_loader = data_processor.create_dataloaders(
            ds_retain_val, ds_forget_val
        )

        logger.info(
            f"Data loaded: {len(retain_train_df)} retain, {len(forget_train_df)} forget examples"
        )

        # Step 2: Load and setup models
        logger.info("=" * 50)
        logger.info("STEP 2: Model Loading and Setup")
        logger.info("=" * 50)

        model, pretrained_model = model_manager.load_models()
        model = model_manager.setup_lora(model)
        tokenizer = model_manager.load_tokenizer()

        # Step 3: Training (Unlearning)
        logger.info("=" * 50)
        logger.info("STEP 3: Unlearning Training")
        logger.info("=" * 50)

        # Extract retain answers for random disassociation
        retain_answers = retain_train_df["output"].tolist()

        # Train unlearned model
        unlearned_model = trainer.train_unlearning(
            model,
            pretrained_model,
            retain_train_loader,
            forget_train_loader,
            tokenizer,
            retain_answers,
        )

        # Save unlearned model
        model_manager.save_model(unlearned_model, config.temp_model_dir)

        # Step 4: Task Vector Creation
        logger.info("=" * 50)
        logger.info("STEP 4: Task Vector Creation and Application")
        logger.info("=" * 50)

        # Reload models in fp32 for task vector computation
        model_manager.cleanup_memory()

        pretrained_model_fp32 = model_manager.load_model_from_path(
            config.pretrained_model_path
        )
        unlearned_model_fp32 = model_manager.load_model_from_path(config.temp_model_dir)

        # Create and apply task vector
        task_vector = TaskVector(pretrained_model_fp32, unlearned_model_fp32)
        neg_task_vector = -task_vector
        final_unlearned_model = neg_task_vector.apply_to(
            pretrained_model_fp32, scaling_coef=config.task_vector_scaling_coef
        )

        # Save final model
        model_manager.save_model(final_unlearned_model, config.task_vector_saving_path)

        # Step 5: Evaluation
        logger.info("=" * 50)
        logger.info("STEP 5: Model Evaluation")
        logger.info("=" * 50)

        # Load models for evaluation
        pretrained_eval_model = model_manager.load_model_from_path(
            config.pretrained_model_path
        )
        unlearned_eval_model = model_manager.load_model_from_path(
            config.task_vector_saving_path
        )

        # Evaluate models
        evaluation_results = evaluator.evaluate_models(
            pretrained_eval_model,
            unlearned_eval_model,
            retain_val_loader,
            forget_val_loader,
            tokenizer,
        )

        # Generate and display samples
        generation_samples = evaluator.generate_samples(
            pretrained_eval_model, unlearned_eval_model, forget_validation_df, tokenizer
        )

        evaluator.print_generation_samples(generation_samples)

        # Save results
        evaluator.save_evaluation_results(evaluation_results, generation_samples)

        logger.info("=" * 50)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 50)

        return evaluation_results, generation_samples

    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
