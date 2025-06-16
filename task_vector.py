"""Task Vector operations for LLM Unlearning project."""

import logging
import torch
from typing import Dict, Optional, Union

logger = logging.getLogger(__name__)


class TaskVector:
    """
    Task Vector implementation for model arithmetic operations.

    A task vector represents the difference between a fine-tuned model and its pretrained base.
    """

    def __init__(
        self,
        pretrained_checkpoint: Optional[torch.nn.Module] = None,
        finetuned_checkpoint: Optional[torch.nn.Module] = None,
        vector: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Initialize the task vector from pretrained and finetuned checkpoints.

        This can either be done by passing two models (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passing in
        the task vector state dict.

        Args:
            pretrained_checkpoint: The pretrained base model
            finetuned_checkpoint: The finetuned model
            vector: Pre-computed task vector state dict
        """
        if vector is not None:
            self.vector = vector
        else:
            assert (
                pretrained_checkpoint is not None and finetuned_checkpoint is not None
            )
            logger.info("Computing task vector from model checkpoints...")

            with torch.no_grad():
                pretrained_state_dict = pretrained_checkpoint.state_dict()
                finetuned_state_dict = finetuned_checkpoint.state_dict()

                self.vector = {}
                for key in pretrained_state_dict:
                    # Skip integer and byte tensors (typically indices or masks)
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue

                    if key in finetuned_state_dict:
                        # Compute the difference (task vector)
                        self.vector[key] = (
                            finetuned_state_dict[key] - pretrained_state_dict[key]
                        )
                    else:
                        logger.warning(f"Key {key} not found in finetuned model")

            logger.info(f"Task vector computed with {len(self.vector)} parameters")

    def __add__(self, other: "TaskVector") -> "TaskVector":
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    logger.warning(f"Key {key} is not present in both task vectors.")
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other: Union[None, int, "TaskVector"]) -> "TaskVector":
        """Support for sum() operations and right addition."""
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self) -> "TaskVector":
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = -self.vector[key]
        return TaskVector(vector=new_vector)

    def __mul__(self, scalar: float) -> "TaskVector":
        """Multiply task vector by a scalar."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = scalar * self.vector[key]
        return TaskVector(vector=new_vector)

    def __rmul__(self, scalar: float) -> "TaskVector":
        """Right multiplication by scalar."""
        return self.__mul__(scalar)

    def apply_to(
        self, pretrained_model: torch.nn.Module, scaling_coef: float = 1.0
    ) -> torch.nn.Module:
        """
        Apply a task vector to a pretrained model.

        Args:
            pretrained_model: The base model to apply the task vector to
            scaling_coef: Scaling coefficient for the task vector

        Returns:
            Model with task vector applied
        """
        logger.info(f"Applying task vector with scaling coefficient {scaling_coef}")

        with torch.no_grad():
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()

            for key in pretrained_state_dict:
                if key not in self.vector:
                    logger.warning(
                        f"Key {key} is present in the pretrained state dict but not in the task vector"
                    )
                    new_state_dict[key] = pretrained_state_dict[key]
                    continue

                # Apply the task vector with scaling
                new_state_dict[key] = (
                    pretrained_state_dict[key] + scaling_coef * self.vector[key]
                )

            # Load the new state dict
            pretrained_model.load_state_dict(new_state_dict, strict=False)

        logger.info("Task vector applied successfully")
        return pretrained_model

    def apply_to_chunked(
        self,
        pretrained_model: torch.nn.Module,
        scaling_coef: float = 1.0,
        chunk_size: int = 500,
    ) -> torch.nn.Module:
        """
        Apply a task vector to a pretrained model in chunks.

        This method is useful when you don't have enough GPU memory to apply
        the task vector in one go.

        Args:
            pretrained_model: The base model to apply the task vector to
            scaling_coef: Scaling coefficient for the task vector
            chunk_size: Number of parameters to process at once

        Returns:
            Model with task vector applied
        """
        logger.info(f"Applying task vector in chunks of size {chunk_size}")

        with torch.no_grad():
            pretrained_state_dict = pretrained_model.state_dict()
            keys = list(self.vector.keys())
            total_keys = len(keys)

            for i in range(0, total_keys, chunk_size):
                new_state_dict = {}
                chunk_keys = keys[i : i + chunk_size]

                for key in chunk_keys:
                    if key not in pretrained_state_dict:
                        logger.warning(
                            f"Key {key} is present in the task vector but not in the pretrained model"
                        )
                        continue

                    # Apply scaling and update the parameter
                    new_state_dict[key] = (
                        pretrained_state_dict[key] + scaling_coef * self.vector[key]
                    )

                # Partially load the updated state dict to the model
                pretrained_model.load_state_dict(new_state_dict, strict=False)

                logger.info(
                    f"Processed chunk {i//chunk_size + 1}/{(total_keys + chunk_size - 1)//chunk_size}"
                )

        logger.info("Chunked task vector application completed")
        return pretrained_model

    def save(self, path: str) -> None:
        """Save task vector to disk."""
        torch.save(self.vector, path)
        logger.info(f"Task vector saved to {path}")

    @classmethod
    def load(cls, path: str) -> "TaskVector":
        """Load task vector from disk."""
        logger.info(f"Loading task vector from {path}")
        vector = torch.load(path, map_location="cpu")
        return cls(vector=vector)

    def get_norm(self) -> float:
        """Get the L2 norm of the task vector."""
        with torch.no_grad():
            total_norm = 0.0
            for key in self.vector:
                param_norm = self.vector[key].norm()
                total_norm += param_norm.item() ** 2
            return total_norm**0.5

    def get_cosine_similarity(self, other: "TaskVector") -> float:
        """Get cosine similarity with another task vector."""
        with torch.no_grad():
            dot_product = 0.0
            norm_self = 0.0
            norm_other = 0.0

            for key in self.vector:
                if key in other.vector:
                    dot_product += torch.sum(
                        self.vector[key] * other.vector[key]
                    ).item()
                    norm_self += torch.sum(self.vector[key] ** 2).item()
                    norm_other += torch.sum(other.vector[key] ** 2).item()

            if norm_self == 0.0 or norm_other == 0.0:
                return 0.0

            return dot_product / (norm_self**0.5 * norm_other**0.5)
