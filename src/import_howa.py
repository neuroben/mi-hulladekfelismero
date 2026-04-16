"""
Import the HOWA dataset into this project's ImageFolder layout.

The HOWA dataset stores one image and one LabelMe JSON annotation per file.
This script reads the polygon annotation, crops a padded bounding box around
the labeled object, maps HOWA labels to this project's classes, and writes the
crop into dataset/<split>/<class>/.

Usage:
    python src/import_howa.py --source_dir "C:\\path\\to\\howa\\dataset" --target_dir dataset
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


SPLITS = ("train", "val", "test")
LABEL_MAP = {
    "glass": "glass",
    "metal": "metal",
    "plastic": "plastic",
    "carton": "paper",
}
IGNORED_LABELS = {"__ignore__", "_background_"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import HOWA crops into dataset/train|val|test/<class>."
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Path to the HOWA dataset root containing train/val/test.",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="dataset",
        help="Target dataset root (default: dataset).",
    )
    parser.add_argument(
        "--padding_ratio",
        type=float,
        default=0.10,
        help="Extra padding around the annotation bbox (default: 0.10).",
    )
    parser.add_argument(
        "--min_padding",
        type=int,
        default=8,
        help="Minimum padding in pixels (default: 8).",
    )
    return parser.parse_args()


def compute_crop_box(points: list[list[float]], width: int, height: int, padding_ratio: float, min_padding: int) -> tuple[int, int, int, int]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    left = min(xs)
    top = min(ys)
    right = max(xs)
    bottom = max(ys)

    bbox_width = max(1.0, right - left)
    bbox_height = max(1.0, bottom - top)
    padding = max(min_padding, int(max(bbox_width, bbox_height) * padding_ratio))

    crop_left = max(0, int(left) - padding)
    crop_top = max(0, int(top) - padding)
    crop_right = min(width, int(right) + padding)
    crop_bottom = min(height, int(bottom) + padding)

    return crop_left, crop_top, crop_right, crop_bottom


def iter_valid_shapes(annotation: dict) -> list[dict]:
    valid_shapes = []
    for shape in annotation.get("shapes", []):
        label = shape.get("label")
        if label in IGNORED_LABELS:
            continue
        if label not in LABEL_MAP:
            continue
        if not shape.get("points"):
            continue
        valid_shapes.append(shape)
    return valid_shapes


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_dir)
    target_root = Path(args.target_dir)

    if not source_root.exists():
        raise FileNotFoundError(f"Source HOWA directory not found: {source_root}")

    imported_counts: dict[str, Counter] = defaultdict(Counter)
    skipped_counts: dict[str, Counter] = defaultdict(Counter)

    for split in SPLITS:
        source_split = source_root / split
        if not source_split.exists():
            print(f"Skipping missing split: {source_split}")
            continue

        for json_path in sorted(source_split.glob("*.json")):
            annotation = json.loads(json_path.read_text(encoding="utf-8"))
            shapes = iter_valid_shapes(annotation)

            if not shapes:
                skipped_counts[split]["no_valid_shape"] += 1
                continue

            image_rel_path = annotation.get("imagePath")
            if not image_rel_path:
                skipped_counts[split]["missing_image_path"] += 1
                continue

            image_path = source_split / image_rel_path
            if not image_path.exists():
                skipped_counts[split]["missing_image_file"] += 1
                continue

            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB")
                width, height = rgb_image.size

                for shape_index, shape in enumerate(shapes):
                    source_label = shape["label"]
                    target_label = LABEL_MAP[source_label]
                    crop_box = compute_crop_box(
                        points=shape["points"],
                        width=width,
                        height=height,
                        padding_ratio=args.padding_ratio,
                        min_padding=args.min_padding,
                    )
                    crop = rgb_image.crop(crop_box)

                    output_dir = target_root / split / target_label
                    output_dir.mkdir(parents=True, exist_ok=True)

                    stem = json_path.stem
                    suffix = f"_{shape_index}" if len(shapes) > 1 else ""
                    output_path = output_dir / f"howa_{stem}{suffix}.png"
                    crop.save(output_path)
                    imported_counts[split][target_label] += 1

    print("HOWA import complete.")
    for split in SPLITS:
        if split in imported_counts:
            total = sum(imported_counts[split].values())
            print(f"{split}: total={total}")
            for label in ("glass", "metal", "paper", "plastic"):
                print(f"  {label}: {imported_counts[split][label]}")
        if split in skipped_counts and skipped_counts[split]:
            print(f"{split} skipped:")
            for reason, count in sorted(skipped_counts[split].items()):
                print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
