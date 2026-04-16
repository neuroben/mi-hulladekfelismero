from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .config import DEFAULT_CLASSES, HOWA_IGNORED_LABELS, HOWA_LABEL_MAP, SPLITS


@dataclass(slots=True)
class HowaImportConfig:
    source_dir: str
    target_dir: str = "dataset"
    padding_ratio: float = 0.10
    min_padding: int = 8


def compute_crop_box(
    points: list[list[float]],
    width: int,
    height: int,
    padding_ratio: float,
    min_padding: int,
) -> tuple[int, int, int, int]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    left = min(xs)
    top = min(ys)
    right = max(xs)
    bottom = max(ys)

    bbox_width = max(1.0, right - left)
    bbox_height = max(1.0, bottom - top)
    padding = max(min_padding, int(max(bbox_width, bbox_height) * padding_ratio))

    return (
        max(0, int(left) - padding),
        max(0, int(top) - padding),
        min(width, int(right) + padding),
        min(height, int(bottom) + padding),
    )


def iter_valid_shapes(annotation: dict) -> list[dict]:
    valid_shapes: list[dict] = []
    for shape in annotation.get("shapes", []):
        label = shape.get("label")
        if label in HOWA_IGNORED_LABELS:
            continue
        if label not in HOWA_LABEL_MAP:
            continue
        if not shape.get("points"):
            continue
        valid_shapes.append(shape)
    return valid_shapes


def import_howa_dataset(
    config: HowaImportConfig,
) -> tuple[dict[str, Counter], dict[str, Counter]]:
    source_root = Path(config.source_dir)
    target_root = Path(config.target_dir)

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
                    target_label = HOWA_LABEL_MAP[shape["label"]]
                    crop_box = compute_crop_box(
                        points=shape["points"],
                        width=width,
                        height=height,
                        padding_ratio=config.padding_ratio,
                        min_padding=config.min_padding,
                    )
                    crop = rgb_image.crop(crop_box)
                    output_dir = target_root / split / target_label
                    output_dir.mkdir(parents=True, exist_ok=True)

                    suffix = f"_{shape_index}" if len(shapes) > 1 else ""
                    output_path = output_dir / f"howa_{json_path.stem}{suffix}.png"
                    crop.save(output_path)
                    imported_counts[split][target_label] += 1

    return imported_counts, skipped_counts


def format_howa_import_report(
    imported_counts: dict[str, Counter],
    skipped_counts: dict[str, Counter],
) -> str:
    lines = ["HOWA import complete."]
    for split in SPLITS:
        if split in imported_counts:
            total = sum(imported_counts[split].values())
            lines.append(f"{split}: total={total}")
            for label in DEFAULT_CLASSES:
                lines.append(f"  {label}: {imported_counts[split][label]}")

        if split in skipped_counts and skipped_counts[split]:
            lines.append(f"{split} skipped:")
            for reason, count in sorted(skipped_counts[split].items()):
                lines.append(f"  {reason}: {count}")

    return "\n".join(lines)
