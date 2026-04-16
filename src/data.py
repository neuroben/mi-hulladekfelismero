"""
data.py – Dataset loading and preprocessing utilities.

Provides:
  - get_transforms()  : torchvision transforms for train / val / test splits
  - get_dataloaders() : builds ImageFolder datasets and wraps them in DataLoaders
"""

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Image size expected by pretrained ResNet models
IMAGE_SIZE = 224

# ImageNet mean and std – required when using pretrained ResNet weights
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(split: str) -> transforms.Compose:
    """
    Return the appropriate torchvision transform pipeline for *split*.

    Training uses random augmentations to reduce over-fitting.
    Validation and test use only the minimal resize + crop + normalise pipeline.

    Args:
        split: One of 'train', 'val', or 'test'.

    Returns:
        A ``transforms.Compose`` instance.
    """
    if split == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(IMAGE_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader, list]:
    """
    Build train, val, and test DataLoaders from an ImageFolder dataset.

    Expected directory layout::

        data_dir/
          train/  <class_a>/  ...
          val/    <class_a>/  ...
          test/   <class_a>/  ...

    Args:
        data_dir:    Root directory containing 'train', 'val', and 'test' sub-dirs.
        batch_size:  Mini-batch size for all loaders.
        num_workers: Number of parallel worker processes for data loading.

    Returns:
        A 4-tuple ``(train_loader, val_loader, test_loader, class_names)``.
    """
    root = Path(data_dir)

    datasets_dict = {
        split: datasets.ImageFolder(
            root=str(root / split),
            transform=get_transforms(split),
        )
        for split in ("train", "val", "test")
    }

    # Verify that all splits share the same classes
    train_classes = datasets_dict["train"].classes
    for split in ("val", "test"):
        assert datasets_dict[split].classes == train_classes, (
            f"Class mismatch between 'train' and '{split}' splits. "
            "Make sure all splits have the same sub-folder names."
        )

    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for split, dataset in datasets_dict.items()
    }

    return loaders["train"], loaders["val"], loaders["test"], train_classes
