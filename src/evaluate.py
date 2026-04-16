"""
evaluate.py – Evaluate a saved model on the test split.

Usage example::

    python src/evaluate.py --model_path models/best_model.pth --data_dir dataset

Prints overall accuracy, a confusion matrix, and per-class precision /
recall / F1 score.
"""

import argparse
import sys
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix

# Allow imports from the src/ package when running as a script
sys.path.insert(0, str(Path(__file__).parent))

from data import get_dataloaders
from model import load_model
from utils import get_device


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the waste classifier on the test set")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the saved model checkpoint (.pth)")
    parser.add_argument("--data_dir", type=str, default="dataset",
                        help="Root directory of the dataset (default: dataset)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Mini-batch size (default: 32)")
    parser.add_argument("--num_classes", type=int, default=4,
                        help="Number of output classes (default: 4)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker processes (default: 4)")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    # --- Data ---
    _, _, test_loader, class_names = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # --- Model ---
    model = load_model(
        checkpoint_path=args.model_path,
        num_classes=args.num_classes,
        device=device,
    )
    print(f"Model loaded from: {args.model_path}")

    # --- Inference on the test set ---
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    # --- Metrics ---
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    overall_acc = correct / len(all_labels)
    print(f"\nTest accuracy: {overall_acc:.4f} ({correct}/{len(all_labels)})")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion matrix (rows = true, cols = predicted):")
    header = "         " + "  ".join(f"{c:>8}" for c in class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>8}" for v in row)
        print(f"{class_names[i]:>8} {row_str}")

    # Per-class precision, recall, F1
    print("\nClassification report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))


if __name__ == "__main__":
    main()
