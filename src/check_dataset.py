"""
Validate the dataset folder structure and print image counts per split/class.

Usage:
    python src/check_dataset.py --data_dir dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_CLASSES = ["glass", "metal", "paper", "plastic"]
SPLITS = ["train", "val", "test"]
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check dataset folders and count images per split/class."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset",
        help="Root dataset directory (default: dataset)",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
        help="Expected class folder names (default: glass metal paper plastic)",
    )
    return parser.parse_args()


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(
        1
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def print_table(counts: dict[str, dict[str, int]], class_names: list[str]) -> None:
    label_width = max(5, max(len(name) for name in class_names))
    split_width = max(5, max(len(split) for split in SPLITS))

    header = [f"{'split':<{split_width}}"]
    header.extend(f"{name:>{label_width}}" for name in class_names)
    header.append(f"{'total':>{label_width}}")
    print("  ".join(header))
    print("  ".join("-" * len(cell) for cell in header))

    for split in SPLITS:
        values = [counts[split][name] for name in class_names]
        row = [f"{split:<{split_width}}"]
        row.extend(f"{value:>{label_width}}" for value in values)
        row.append(f"{sum(values):>{label_width}}")
        print("  ".join(row))


def main() -> None:
    args = parse_args()
    root = Path(args.data_dir)
    class_names = args.classes

    if not root.exists():
        print(f"ERROR: dataset root not found: {root}")
        sys.exit(1)

    counts: dict[str, dict[str, int]] = {split: {} for split in SPLITS}
    missing_dirs: list[Path] = []
    empty_class_dirs: list[Path] = []
    extra_class_dirs: list[Path] = []

    for split in SPLITS:
        split_dir = root / split
        if not split_dir.exists():
            missing_dirs.append(split_dir)
            for class_name in class_names:
                counts[split][class_name] = 0
            continue

        actual_class_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
        actual_class_names = {path.name for path in actual_class_dirs}

        for extra_name in sorted(actual_class_names - set(class_names)):
            extra_class_dirs.append(split_dir / extra_name)

        for class_name in class_names:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                missing_dirs.append(class_dir)
                counts[split][class_name] = 0
                continue

            image_count = count_images(class_dir)
            counts[split][class_name] = image_count
            if image_count == 0:
                empty_class_dirs.append(class_dir)

    print(f"Dataset root: {root.resolve()}")
    print(f"Expected classes: {', '.join(class_names)}")
    print("")
    print_table(counts, class_names)

    print("")
    total_images = 0
    for split in SPLITS:
        split_total = sum(counts[split].values())
        total_images += split_total
        print(f"{split:>5} total: {split_total}")
    print(f"overall total: {total_images}")

    if missing_dirs:
        print("")
        print("Missing folders:")
        for path in missing_dirs:
            print(f"  - {path}")

    if empty_class_dirs:
        print("")
        print("Empty class folders:")
        for path in empty_class_dirs:
            print(f"  - {path}")

    if extra_class_dirs:
        print("")
        print("Unexpected class folders:")
        for path in extra_class_dirs:
            print(f"  - {path}")

    class_totals = {
        class_name: sum(counts[split][class_name] for split in SPLITS)
        for class_name in class_names
    }
    non_zero_totals = [count for count in class_totals.values() if count > 0]

    if non_zero_totals:
        min_count = min(non_zero_totals)
        max_count = max(non_zero_totals)
        if min_count > 0 and max_count / min_count >= 2:
            print("")
            print("Warning: the dataset looks imbalanced.")
            for class_name in class_names:
                print(f"  - {class_name}: {class_totals[class_name]} images")

    has_errors = bool(missing_dirs or empty_class_dirs or total_images == 0)
    print("")
    if has_errors:
        print("Result: dataset is not ready for training yet.")
        sys.exit(1)

    print("Result: dataset structure looks valid.")


if __name__ == "__main__":
    main()
