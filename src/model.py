"""
model.py – Model definition and loading utilities.

Provides:
  - build_model()   : loads a pretrained ResNet18 and replaces the head
  - freeze_backbone() / unfreeze_backbone() : helpers for transfer learning
"""

import torch
import torch.nn as nn
from typing import Optional
from torchvision import models
from torchvision.models import ResNet18_Weights


def build_model(
    num_classes: int = 4,
    freeze_backbone: bool = True,
    pretrained: bool = True,
) -> nn.Module:
    """
    Build a ResNet18-based classifier.

    The backbone is loaded with ImageNet weights by default. The final fully
    connected layer is replaced with a new ``Linear(512, num_classes)`` layer
    whose weights are randomly initialised.

    Args:
        num_classes:       Number of output categories.
        freeze_backbone:   If ``True``, all backbone parameters are frozen so
                           only the new head is trained (Stage 1 transfer learning).
        pretrained:        Load ImageNet weights for the backbone.

    Returns:
        A ``torch.nn.Module`` ready for training.
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final FC layer (512 → num_classes)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    # The new head's parameters always require gradients
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def unfreeze_backbone(model: nn.Module) -> None:
    """
    Unfreeze all backbone parameters so the entire network is fine-tuned.

    Call this after an initial frozen-backbone training phase (Stage 2).
    Remember to lower the learning rate before continuing training.

    Args:
        model: The ResNet model returned by :func:`build_model`.
    """
    for param in model.parameters():
        param.requires_grad = True


def load_model(
    checkpoint_path: str,
    num_classes: int = 4,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Restore a model from a checkpoint file saved by ``train.py``.

    Args:
        checkpoint_path: Path to the ``.pth`` file.
        num_classes:     Number of output classes (must match checkpoint).
        device:          Target device. Defaults to CPU when ``None``.

    Returns:
        The loaded model in evaluation mode.
    """
    if device is None:
        device = torch.device("cpu")

    # Build the architecture first (backbone can be unfrozen since we are
    # loading weights anyway)
    model = build_model(num_classes=num_classes, freeze_backbone=False, pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
