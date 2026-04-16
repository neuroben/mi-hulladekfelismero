from __future__ import annotations

import argparse

from waste_classifier.training import TrainConfig, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the waste classifier")
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint_dir", type=str, default="models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unfreeze_all", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_model(
        TrainConfig(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_classes=args.num_classes,
            num_workers=args.num_workers,
            checkpoint_dir=args.checkpoint_dir,
            seed=args.seed,
            unfreeze_all=args.unfreeze_all,
        )
    )


if __name__ == "__main__":
    main()
