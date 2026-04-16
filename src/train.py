"""
train.py – Training script for the waste classifier.

Usage example::

    python src/train.py --data_dir dataset --epochs 15 --batch_size 32

All configuration values can be set via CLI arguments (see --help).
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import Adam

# Allow imports from the src/ package when running as a script
sys.path.insert(0, str(Path(__file__).parent))

from data import get_dataloaders
from model import build_model, unfreeze_backbone
from utils import accuracy, get_device, set_seed


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the waste classifier")

    parser.add_argument("--data_dir", type=str, default="dataset",
                        help="Root directory of the dataset (default: dataset)")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of training epochs (default: 15)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Mini-batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for Adam (default: 1e-3)")
    parser.add_argument("--num_classes", type=int, default=4,
                        help="Number of output classes (default: 4)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker processes (default: 4)")
    parser.add_argument("--checkpoint_dir", type=str, default="models",
                        help="Directory to save the best model (default: models)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--unfreeze_all", action="store_true",
                        help="Unfreeze all backbone layers (full fine-tuning)")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training / validation helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, avg_accuracy)."""
    model.train()
    total_loss = 0.0
    total_acc = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy(outputs, labels)

    n = len(loader)
    return total_loss / n, total_acc / n


def evaluate_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Run one evaluation pass. Returns (avg_loss, avg_accuracy)."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_acc += accuracy(outputs, labels)

    n = len(loader)
    return total_loss / n, total_acc / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    # --- Data ---
    train_loader, val_loader, _, class_names = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Classes ({len(class_names)}): {class_names}")

    # --- Model ---
    freeze = not args.unfreeze_all
    model = build_model(num_classes=args.num_classes, freeze_backbone=freeze)
    model.to(device)

    if args.unfreeze_all:
        unfreeze_backbone(model)

    # --- Loss & optimiser ---
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )

    # --- Checkpoint directory ---
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = checkpoint_dir / "best_model.pth"

    # --- Training loop ---
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_one_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch:>3}/{args.epochs}]  "
            f"Train loss: {train_loss:.4f}  Train acc: {train_acc:.4f}  "
            f"Val loss: {val_loss:.4f}  Val acc: {val_acc:.4f}"
        )

        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_checkpoint)
            print(f"  ✓ Best model saved (val_acc={val_acc:.4f})")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Best checkpoint saved to: {best_checkpoint}")


if __name__ == "__main__":
    main()
