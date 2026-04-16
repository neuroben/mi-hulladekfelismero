"""
predict.py – Single-image inference with a saved model.

Usage example::

    python src/predict.py --model_path models/best_model.pth --image_path photo.jpg

Prints the predicted class label and the softmax probability for each class.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

# Allow imports from the src/ package when running as a script
sys.path.insert(0, str(Path(__file__).parent))

from data import get_transforms
from model import load_model
from utils import get_device


# Default class names (alphabetical ImageFolder order).
# Edit this list if your dataset uses different class names.
DEFAULT_CLASSES = ["glass", "metal", "paper", "plastic"]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict the waste type of a single image")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the saved model checkpoint (.pth)")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the input image")
    parser.add_argument("--num_classes", type=int, default=4,
                        help="Number of output classes (default: 4)")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES,
                        help="Class names in the same order as training folders")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Prediction helper (also used by app.py)
# ---------------------------------------------------------------------------

def predict_image(
    image: Image.Image,
    model: torch.nn.Module,
    class_names: List[str],
    device: torch.device,
) -> Tuple[str, Dict[str, float]]:
    """
    Run inference on a single PIL image.

    Args:
        image:       Input PIL image (RGB).
        model:       Loaded model in eval mode.
        class_names: Ordered list of class name strings.
        device:      Target device.

    Returns:
        A 2-tuple ``(predicted_class, {class_name: probability, ...})``.
    """
    transform = get_transforms("test")
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu()

    predicted_idx = probs.argmax().item()
    predicted_class = class_names[predicted_idx]
    prob_dict = {name: round(probs[i].item(), 4) for i, name in enumerate(class_names)}

    return predicted_class, prob_dict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = get_device()

    # Load model
    model = load_model(
        checkpoint_path=args.model_path,
        num_classes=args.num_classes,
        device=device,
    )

    # Load image
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: image not found at '{image_path}'")
        sys.exit(1)
    image = Image.open(image_path)

    # Predict
    predicted_class, prob_dict = predict_image(image, model, args.classes, device)

    print(f"Predicted class : {predicted_class}")
    print("Class probabilities:")
    for cls, prob in sorted(prob_dict.items(), key=lambda x: -x[1]):
        bar = "#" * int(prob * 30)
        print(f"  {cls:>10}: {prob:.4f}  {bar}")


if __name__ == "__main__":
    main()
