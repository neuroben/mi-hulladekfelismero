from __future__ import annotations

import argparse
import sys

from waste_classifier.config import DEFAULT_CLASSES
from waste_classifier.gui import launch_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Waste classifier Gradio demo")
    parser.add_argument("--model_path", type=str, default="models/best_model.pth")
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        launch_app(
            model_path=args.model_path,
            num_classes=args.num_classes,
            class_names=args.classes,
            port=args.port,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        print("Train the model first with: python scripts/train.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
