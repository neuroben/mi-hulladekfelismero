from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def build_model(
    num_classes: int = 4,
    freeze_backbone: bool = True,
    pretrained: bool = True,
) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def unfreeze_backbone(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def load_model(
    checkpoint_path: str,
    num_classes: int = 4,
    device: torch.device | None = None,
) -> nn.Module:
    target_device = device or torch.device("cpu")
    model = build_model(num_classes=num_classes, freeze_backbone=False, pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=target_device)
    model.load_state_dict(state_dict)
    model.to(target_device)
    model.eval()
    return model
