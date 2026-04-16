from __future__ import annotations

import argparse

from waste_classifier.evaluation import EvaluationConfig, evaluate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the waste classifier on the test set")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_model(
        EvaluationConfig(
            model_path=args.model_path,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_classes=args.num_classes,
            num_workers=args.num_workers,
        )
    )


if __name__ == "__main__":
    main()
