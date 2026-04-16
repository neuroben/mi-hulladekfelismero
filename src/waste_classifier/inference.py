from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from PIL import Image

from .data import get_transforms


def predict_image(
    image: Image.Image,
    model: torch.nn.Module,
    class_names: list[str],
    device: torch.device,
) -> tuple[str, Dict[str, float]]:
    transform = get_transforms("test")
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0).cpu()

    predicted_index = probabilities.argmax().item()
    predicted_class = class_names[predicted_index]
    probability_map = {
        name: round(probabilities[index].item(), 4)
        for index, name in enumerate(class_names)
    }
    return predicted_class, probability_map
