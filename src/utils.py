"""
utils.py – Shared helper functions used across the project.
"""

import random

import numpy as np
import torch


def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    """
    Fix all random seeds for reproducibility.

    Sets seeds for Python's ``random``, NumPy, and PyTorch (CPU & GPU).

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute classification accuracy for a single batch.

    Args:
        outputs: Raw model logits of shape ``(N, C)``.
        labels:  Ground-truth class indices of shape ``(N,)``.

    Returns:
        Fraction of correctly classified samples (float in [0, 1]).
    """
    _, predicted = torch.max(outputs, dim=1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)
