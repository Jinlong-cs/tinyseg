from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml


def build_parser():
    parser = argparse.ArgumentParser(description="Merge multiple YOLO segmentation datasets into one dataset root.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input YOLO dataset roots.")
    parser.add_argument("--output", required=True, help="Output merged dataset root.")
    parser.add_argument(
        "--prefix-mode",
        choices=["dirname", "index"],
        default="dirname",
        help="How to prefix copied file names to avoid collisions.",
    )
    return parser


def read_data_yaml(dataset_root: Path):
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.is_file():
        raise SystemExit(f"missing data.yaml: {data_yaml}")

    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = data.get("names")
    if isinstance(names, dict):
        names = [names[index] for index in sorted(names)]

    return {
        "path": dataset_root,
        "train": data["train"],
        "val": data["val"],
        "nc": int(data["nc"]),
        "names": list(names),
    }


def ensure_layout(output_root: Path):
    for relative in [
        "images/train",
        "images/val",
        "labels/train",
        "labels/val",
    ]:
        (output_root / relative).mkdir(parents=True, exist_ok=True)


def iter_split_pairs(dataset_root: Path, split_source: str):
    images_dir = dataset_root / split_source
    labels_dir = dataset_root / split_source.replace("images", "labels", 1)
    if not images_dir.is_dir():
        raise SystemExit(f"missing images dir: {images_dir}")
    if not labels_dir.is_dir():
        raise SystemExit(f"missing labels dir: {labels_dir}")

    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file():
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise SystemExit(f"missing label for image: {image_path}")
        yield image_path, label_path


def copy_split(dataset_index: int, dataset_root: Path, split_name: str, split_source: str, output_root: Path, prefix_mode: str):
    if prefix_mode == "dirname":
        prefix = dataset_root.name
    else:
        prefix = f"ds{dataset_index:02d}"

    image_count = 0
    for image_path, label_path in iter_split_pairs(dataset_root, split_source):
        dst_stem = f"{prefix}_{image_path.stem}"
        dst_image = output_root / "images" / split_name / f"{dst_stem}{image_path.suffix.lower()}"
        dst_label = output_root / "labels" / split_name / f"{dst_stem}.txt"
        if dst_image.exists() or dst_label.exists():
            raise SystemExit(f"duplicate destination stem: {dst_stem}")
        shutil.copy2(image_path, dst_image)
        shutil.copy2(label_path, dst_label)
        image_count += 1
    return image_count


def write_data_yaml(output_root: Path, nc: int, names: list[str]):
    data = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": nc,
        "names": {index: name for index, name in enumerate(names)},
    }
    with (output_root / "data.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=False)


def run_merge(args):
    output_root = Path(args.output).resolve()
    inputs = [Path(item).resolve() for item in args.inputs]

    ensure_layout(output_root)

    dataset_infos = [read_data_yaml(path) for path in inputs]
    if not dataset_infos:
        raise SystemExit("no datasets were provided")

    reference_names = dataset_infos[0]["names"]
    reference_nc = dataset_infos[0]["nc"]
    for info in dataset_infos[1:]:
        if info["nc"] != reference_nc:
            raise SystemExit(f"dataset nc mismatch: {info['path']}")
        if info["names"] != reference_names:
            raise SystemExit(f"dataset names mismatch: {info['path']}")

    summary = {
        "datasets": [],
        "merged": {
            "train_images": 0,
            "val_images": 0,
        },
    }

    for output_dir in [output_root / "images" / "train", output_root / "images" / "val", output_root / "labels" / "train", output_root / "labels" / "val"]:
        for path in output_dir.iterdir():
            if path.is_file():
                path.unlink()

    for dataset_index, info in enumerate(dataset_infos, start=1):
        train_count = copy_split(
            dataset_index=dataset_index,
            dataset_root=info["path"],
            split_name="train",
            split_source=info["train"],
            output_root=output_root,
            prefix_mode=args.prefix_mode,
        )
        val_count = copy_split(
            dataset_index=dataset_index,
            dataset_root=info["path"],
            split_name="val",
            split_source=info["val"],
            output_root=output_root,
            prefix_mode=args.prefix_mode,
        )
        summary["datasets"].append(
            {
                "name": info["path"].name,
                "path": str(info["path"]),
                "train_images": train_count,
                "val_images": val_count,
            }
        )
        summary["merged"]["train_images"] += train_count
        summary["merged"]["val_images"] += val_count

    summary["merged"]["images_total"] = summary["merged"]["train_images"] + summary["merged"]["val_images"]
    summary["merged"]["nc"] = reference_nc
    summary["merged"]["names"] = reference_names

    write_data_yaml(output_root=output_root, nc=reference_nc, names=reference_names)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Merged dataset: {output_root}")
    print(f"Train images:   {summary['merged']['train_images']}")
    print(f"Val images:     {summary['merged']['val_images']}")
    print(f"Summary:        {output_root / 'summary.json'}")
    print(f"Data yaml:      {output_root / 'data.yaml'}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_merge(args)


if __name__ == "__main__":
    main()
