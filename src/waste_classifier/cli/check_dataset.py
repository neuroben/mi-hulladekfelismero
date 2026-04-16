from __future__ import annotations

import argparse
import sys

from waste_classifier.config import DEFAULT_CLASSES
from waste_classifier.dataset_tools import format_dataset_report, inspect_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check dataset folders and count images per split/class."
    )
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = inspect_dataset(data_dir=args.data_dir, class_names=args.classes)
    print(format_dataset_report(report))
    if report.has_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
