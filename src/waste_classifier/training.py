from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .data import get_dataloaders
from .model import build_model
from .utils import accuracy, get_device, set_seed


@dataclass(slots=True)
class TrainConfig:
    data_dir: str = "dataset"
    epochs: int = 15
    batch_size: int = 32
    lr: float = 1e-3
    num_classes: int = 4
    num_workers: int = 4
    checkpoint_dir: str = "models"
    seed: int = 42
    unfreeze_all: bool = False


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy(outputs, labels)

    num_batches = len(loader)
    return total_loss / num_batches, total_acc / num_batches


def evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_acc += accuracy(outputs, labels)

    num_batches = len(loader)
    return total_loss / num_batches, total_acc / num_batches


def train_model(config: TrainConfig) -> Path:
    set_seed(config.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, _, class_names = get_dataloaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    print(f"Classes ({len(class_names)}): {class_names}")

    model = build_model(
        num_classes=config.num_classes,
        freeze_backbone=not config.unfreeze_all,
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.lr,
    )

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = checkpoint_dir / "best_model.pth"
    best_val_acc = -1.0

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_acc = evaluate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Epoch [{epoch:>3}/{config.epochs}]  "
            f"Train loss: {train_loss:.4f}  Train acc: {train_acc:.4f}  "
            f"Val loss: {val_loss:.4f}  Val acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_checkpoint)
            print(f"  [OK] Best model saved (val_acc={val_acc:.4f})")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Best checkpoint saved to: {best_checkpoint}")
    return best_checkpoint
