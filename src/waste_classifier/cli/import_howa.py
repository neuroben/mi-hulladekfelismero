from __future__ import annotations

import argparse

from waste_classifier.howa import HowaImportConfig, format_howa_import_report, import_howa_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import HOWA crops into dataset/train|val|test/<class>."
    )
    parser.add_argument("--source_dir", type=str, required=True)
    parser.add_argument("--target_dir", type=str, default="dataset")
    parser.add_argument("--padding_ratio", type=float, default=0.10)
    parser.add_argument("--min_padding", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    imported_counts, skipped_counts = import_howa_dataset(
        HowaImportConfig(
            source_dir=args.source_dir,
            target_dir=args.target_dir,
            padding_ratio=args.padding_ratio,
            min_padding=args.min_padding,
        )
    )
    print(format_howa_import_report(imported_counts, skipped_counts))


if __name__ == "__main__":
    main()
