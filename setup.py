#!/usr/bin/env python3
"""Setup script for LLM Unlearning project."""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, check=True):
    """Run a shell command."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=check)
    return result


def install_dependencies():
    """Install project dependencies."""
    print("Installing dependencies...")
    run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def setup_environment():
    """Setup the environment for the project."""
    print("Setting up environment...")

    # Create necessary directories
    directories = ["logs", "outputs", "checkpoints", "train", "validation"]

    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"Created directory: {directory}")


def check_cuda():
    """Check if CUDA is available."""
    try:
        import torch

        if torch.cuda.is_available():
            print(f"CUDA is available. Device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("CUDA is not available. Training will use CPU.")
    except ImportError:
        print("PyTorch not installed yet. Run setup first.")


def validate_setup():
    """Validate the setup by importing key modules."""
    print("Validating setup...")

    try:
        # Test imports
        import torch
        import transformers
        import datasets
        import pandas as pd
        import accelerate
        import peft

        print("✓ All required packages are installed successfully!")

        # Check versions
        print(f"PyTorch version: {torch.__version__}")
        print(f"Transformers version: {transformers.__version__}")
        print(f"Datasets version: {datasets.__version__}")

        return True

    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Please run 'python setup.py install' first.")
        return False


def run_quick_test():
    """Run a quick test to ensure everything works."""
    print("Running quick test...")

    try:
        from config import Config
        from data_utils import DataProcessor
        from model_utils import ModelManager

        # Test configuration
        config = Config()
        print("✓ Configuration loaded successfully!")

        # Test data processor initialization
        data_processor = DataProcessor(config)
        print("✓ Data processor initialized successfully!")

        # Test model manager initialization
        model_manager = ModelManager(config)
        print("✓ Model manager initialized successfully!")

        print("✓ Quick test passed!")
        return True

    except Exception as e:
        print(f"✗ Quick test failed: {e}")
        return False


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description="Setup script for LLM Unlearning project"
    )
    parser.add_argument(
        "action",
        choices=["install", "setup", "check", "validate", "test"],
        help="Action to perform",
    )

    args = parser.parse_args()

    if args.action == "install":
        install_dependencies()
    elif args.action == "setup":
        setup_environment()
        print("Environment setup complete!")
    elif args.action == "check":
        check_cuda()
    elif args.action == "validate":
        validate_setup()
    elif args.action == "test":
        if validate_setup():
            run_quick_test()

    print("Setup script completed!")


if __name__ == "__main__":
    main()
