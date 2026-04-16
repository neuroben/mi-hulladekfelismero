"""
app.py – Lightweight Gradio web demo for the waste classifier.

Usage example::

    python app/app.py --model_path models/best_model.pth

Then open http://localhost:7860 in your browser.
"""

import argparse
import sys
from pathlib import Path

import gradio as gr
import torch
from PIL import Image

# Allow importing from src/ when running from the project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import load_model
from predict import predict_image
from utils import get_device


# Default class names – should match your training data folder names
DEFAULT_CLASSES = ["glass", "metal", "paper", "plastic"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Waste classifier Gradio demo")
    parser.add_argument("--model_path", type=str, default="models/best_model.pth",
                        help="Path to the saved model checkpoint (default: models/best_model.pth)")
    parser.add_argument("--num_classes", type=int, default=4,
                        help="Number of output classes (default: 4)")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES,
                        help="Class names in training folder order")
    parser.add_argument("--port", type=int, default=7860,
                        help="Port to run the Gradio server on (default: 7860)")
    return parser.parse_args()


def build_interface(model: torch.nn.Module, class_names: list[str], device: torch.device) -> gr.Interface:
    """Construct and return the Gradio Interface."""

    def classify(image: Image.Image):
        """Gradio prediction callback."""
        if image is None:
            return "No image provided.", {}
        predicted_class, prob_dict = predict_image(image, model, class_names, device)
        # Gradio Label component expects {label: confidence} dict
        return predicted_class, prob_dict

    interface = gr.Interface(
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
        examples=[],   # Add example image paths here if available
        flagging_mode="never",
    )
    return interface


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: model checkpoint not found at '{model_path}'")
        print("Train the model first with:  python src/train.py")
        sys.exit(1)

    model = load_model(
        checkpoint_path=str(model_path),
        num_classes=args.num_classes,
        device=device,
    )
    print(f"Model loaded from: {model_path}")

    interface = build_interface(model, args.classes, device)
    interface.launch(server_port=args.port)


if __name__ == "__main__":
    main()
