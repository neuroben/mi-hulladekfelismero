from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import DEFAULT_CLASSES, IMAGE_EXTENSIONS, SPLITS


@dataclass(slots=True)
class DatasetReport:
    root: Path
    class_names: list[str]
    counts: dict[str, dict[str, int]]
    missing_dirs: list[Path]
    empty_class_dirs: list[Path]
    extra_class_dirs: list[Path]

    @property
    def total_images(self) -> int:
        return sum(sum(split_counts.values()) for split_counts in self.counts.values())

    @property
    def class_totals(self) -> dict[str, int]:
        return {
            class_name: sum(self.counts[split][class_name] for split in SPLITS)
            for class_name in self.class_names
        }

    @property
    def has_errors(self) -> bool:
        return bool(self.missing_dirs or self.empty_class_dirs or self.total_images == 0)

    @property
    def is_imbalanced(self) -> bool:
        non_zero_totals = [count for count in self.class_totals.values() if count > 0]
        if not non_zero_totals:
            return False
        return max(non_zero_totals) / min(non_zero_totals) >= 2


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(
        1
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def inspect_dataset(
    data_dir: str,
    class_names: list[str] | None = None,
) -> DatasetReport:
    root = Path(data_dir)
    expected_classes = class_names or list(DEFAULT_CLASSES)
    counts: dict[str, dict[str, int]] = {split: {} for split in SPLITS}
    missing_dirs: list[Path] = []
    empty_class_dirs: list[Path] = []
    extra_class_dirs: list[Path] = []

    for split in SPLITS:
        split_dir = root / split
        if not split_dir.exists():
            missing_dirs.append(split_dir)
            for class_name in expected_classes:
                counts[split][class_name] = 0
            continue

        actual_class_names = {path.name for path in split_dir.iterdir() if path.is_dir()}
        for extra_name in sorted(actual_class_names - set(expected_classes)):
            extra_class_dirs.append(split_dir / extra_name)

        for class_name in expected_classes:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                missing_dirs.append(class_dir)
                counts[split][class_name] = 0
                continue

            image_count = count_images(class_dir)
            counts[split][class_name] = image_count
            if image_count == 0:
                empty_class_dirs.append(class_dir)

    return DatasetReport(
        root=root,
        class_names=expected_classes,
        counts=counts,
        missing_dirs=missing_dirs,
        empty_class_dirs=empty_class_dirs,
        extra_class_dirs=extra_class_dirs,
    )


def format_dataset_report(report: DatasetReport) -> str:
    label_width = max(5, max(len(name) for name in report.class_names))
    split_width = max(5, max(len(split) for split in SPLITS))

    header = [f"{'split':<{split_width}}"]
    header.extend(f"{name:>{label_width}}" for name in report.class_names)
    header.append(f"{'total':>{label_width}}")

    lines = [
        f"Dataset root: {report.root.resolve()}",
        f"Expected classes: {', '.join(report.class_names)}",
        "",
        "  ".join(header),
        "  ".join("-" * len(cell) for cell in header),
    ]

    for split in SPLITS:
        values = [report.counts[split][name] for name in report.class_names]
        row = [f"{split:<{split_width}}"]
        row.extend(f"{value:>{label_width}}" for value in values)
        row.append(f"{sum(values):>{label_width}}")
        lines.append("  ".join(row))

    lines.append("")
    for split in SPLITS:
        lines.append(f"{split:>5} total: {sum(report.counts[split].values())}")
    lines.append(f"overall total: {report.total_images}")

    if report.missing_dirs:
        lines.extend(["", "Missing folders:"])
        lines.extend(f"  - {path}" for path in report.missing_dirs)

    if report.empty_class_dirs:
        lines.extend(["", "Empty class folders:"])
        lines.extend(f"  - {path}" for path in report.empty_class_dirs)

    if report.extra_class_dirs:
        lines.extend(["", "Unexpected class folders:"])
        lines.extend(f"  - {path}" for path in report.extra_class_dirs)

    if report.is_imbalanced:
        lines.extend(["", "Warning: the dataset looks imbalanced."])
        lines.extend(
            f"  - {class_name}: {report.class_totals[class_name]} images"
            for class_name in report.class_names
        )

    lines.append("")
    if report.has_errors:
        lines.append("Result: dataset is not ready for training yet.")
    else:
        lines.append("Result: dataset structure looks valid.")

    return "\n".join(lines)
