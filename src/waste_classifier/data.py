from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def get_transforms(split: str) -> transforms.Compose:
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
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    root = Path(data_dir)
    datasets_by_split = {
        split: datasets.ImageFolder(
            root=str(root / split),
            transform=get_transforms(split),
        )
        for split in ("train", "val", "test")
    }

    train_classes = datasets_by_split["train"].classes
    for split in ("val", "test"):
        if datasets_by_split[split].classes != train_classes:
            raise ValueError(
                f"Class mismatch between 'train' and '{split}'. "
                "All splits must use the same class folders."
            )

    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for split, dataset in datasets_by_split.items()
    }

    return loaders["train"], loaders["val"], loaders["test"], train_classes
