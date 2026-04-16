from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from waste_classifier.config import DEFAULT_CLASSES
from waste_classifier.inference import predict_image
from waste_classifier.model import load_model
from waste_classifier.utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict the waste type of a single image")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    model = load_model(
        checkpoint_path=args.model_path,
        num_classes=args.num_classes,
        device=device,
    )

    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: image not found at '{image_path}'")
        sys.exit(1)

    with Image.open(image_path) as image:
        predicted_class, probability_map = predict_image(image, model, args.classes, device)

    print(f"Predicted class : {predicted_class}")
    print("Class probabilities:")
    for class_name, probability in sorted(probability_map.items(), key=lambda item: -item[1]):
        bar = "#" * int(probability * 30)
        print(f"  {class_name:>10}: {probability:.4f}  {bar}")


if __name__ == "__main__":
    main()
