from __future__ import annotations

from dataclasses import dataclass

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from .data import get_dataloaders
from .model import load_model
from .utils import get_device


@dataclass(slots=True)
class EvaluationConfig:
    model_path: str
    data_dir: str = "dataset"
    batch_size: int = 32
    num_classes: int = 4
    num_workers: int = 4


def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    predictions: list[int] = []
    labels_all: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            predictions.extend(outputs.argmax(dim=1).cpu().tolist())
            labels_all.extend(labels.tolist())

    return predictions, labels_all


def build_evaluation_report(
    labels: list[int],
    predictions: list[int],
    class_names: list[str],
) -> str:
    correct = sum(prediction == label for prediction, label in zip(predictions, labels))
    total = len(labels)
    accuracy_value = correct / total
    matrix = confusion_matrix(labels, predictions)
    header = "         " + "  ".join(f"{name:>8}" for name in class_names)

    lines = [f"Test accuracy: {accuracy_value:.4f} ({correct}/{total})", ""]
    lines.append("Confusion matrix (rows = true, cols = predicted):")
    lines.append(header)
    for index, row in enumerate(matrix):
        row_values = "  ".join(f"{value:>8}" for value in row)
        lines.append(f"{class_names[index]:>8} {row_values}")

    lines.extend(
        [
            "",
            "Classification report:",
            classification_report(
                labels,
                predictions,
                target_names=class_names,
                zero_division=0,
            ).rstrip(),
        ]
    )
    return "\n".join(lines)


def evaluate_model(config: EvaluationConfig) -> str:
    device = get_device()
    print(f"Using device: {device}")

    _, _, test_loader, class_names = get_dataloaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    model = load_model(
        checkpoint_path=config.model_path,
        num_classes=config.num_classes,
        device=device,
    )
    print(f"Model loaded from: {config.model_path}")

    predictions, labels = collect_predictions(model, test_loader, device)
    report = build_evaluation_report(labels, predictions, class_names)
    print("")
    print(report)
    return report
