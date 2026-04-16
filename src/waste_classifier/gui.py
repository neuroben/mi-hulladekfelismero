from __future__ import annotations

from pathlib import Path

import gradio as gr
import torch
from PIL import Image

from .inference import predict_image
from .model import load_model
from .utils import get_device


def build_interface(
    model: torch.nn.Module,
    class_names: list[str],
    device: torch.device,
) -> gr.Interface:
    def classify(image: Image.Image):
        if image is None:
            return "No image provided.", {}
        predicted_class, probability_map = predict_image(image, model, class_names, device)
        return predicted_class, probability_map

    return gr.Interface(
        fn=classify,
        inputs=gr.Image(type="pil", label="Upload a waste image"),
        outputs=[
            gr.Textbox(label="Predicted class"),
            gr.Label(num_top_classes=len(class_names), label="Class probabilities"),
        ],
        title="Waste Classifier",
        description=(
            "Upload an image of waste and the model will predict whether it belongs to: "
            + ", ".join(class_names)
            + "."
        ),
        examples=[],
        flagging_mode="never",
    )


def launch_app(
    model_path: str,
    num_classes: int,
    class_names: list[str],
    port: int,
) -> None:
    device = get_device()
    print(f"Using device: {device}")

    checkpoint_path = Path(model_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    model = load_model(
        checkpoint_path=str(checkpoint_path),
        num_classes=num_classes,
        device=device,
    )
    print(f"Model loaded from: {checkpoint_path}")

    interface = build_interface(model, class_names, device)
    interface.launch(server_port=port)
