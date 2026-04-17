from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_NAMES = ["drivable", "stairs"]

LABEL_TO_CLASS_ID = {
    "drivable": 0,
    "drivable_area": 0,
    "free_traversable": 0,
    "cautious_traversable": 0,
    "roadway_nonped": 0,
    "ground": 0,
    "floor": 0,
    "地面": 0,
    "stairs": 1,
    "stairs额": 1,
    "stair": 1,
    "stairs_escalator": 1,
    "楼梯": 1,
}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Convert Labelme annotations to a 2-class YOLO-seg dataset: drivable and stairs."
    )
    parser.add_argument("--inputs", nargs="+", required=True, help="Input directories containing Labelme JSON files.")
    parser.add_argument("--output", required=True, help="Output YOLO dataset root.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument(
        "--split-mode",
        choices=["temporal", "random"],
        default="temporal",
        help="How to create the train/val split from sorted samples.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split-mode=random.")
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=["labelme", ".git", "__pycache__"],
        help="Directory name to skip while scanning. Can be provided multiple times.",
    )
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="Drop images that contain no drivable/stairs polygons after label filtering.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output dataset directory.",
    )
    return parser


def normalize_label(label_name: str) -> int | None:
    normalized = label_name.strip().lower()
    return LABEL_TO_CLASS_ID.get(normalized)


def should_skip(path: Path, excluded_dir_names: set[str]) -> bool:
    return any(part in excluded_dir_names for part in path.parts)


def find_image_path(json_path: Path, image_name: str | None) -> Path | None:
    candidates = []
    if image_name:
        candidates.append(json_path.parent / image_name)
    for extension in IMAGE_EXTENSIONS:
        candidates.append(json_path.with_suffix(extension))

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def read_image_size(image_path: Path) -> tuple[int, int]:
    import cv2

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"unable to read image: {image_path}")
    height, width = image.shape[:2]
    return width, height


def safe_stem(source_root: Path, json_path: Path) -> str:
    relative = json_path.relative_to(source_root)
    raw = "_".join(relative.with_suffix("").parts)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def parse_labelme_json(json_path: Path, source_root: Path) -> tuple[dict[str, Any] | None, Counter, str | None]:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, Counter(), f"{json_path}: {exc}"

    if "shapes" not in data:
        return None, Counter(), None

    image_path = find_image_path(json_path, data.get("imagePath"))
    if image_path is None:
        return None, Counter(), f"missing image for {json_path}"

    image_width = data.get("imageWidth")
    image_height = data.get("imageHeight")
    if not image_width or not image_height:
        image_width, image_height = read_image_size(image_path)
    image_width = int(image_width)
    image_height = int(image_height)

    yolo_lines = []
    polygon_labels = []
    ignored_labels = Counter()

    for shape in data.get("shapes") or []:
        if shape.get("shape_type", "polygon") != "polygon":
            continue

        raw_label = str(shape.get("label", "")).strip()
        class_id = normalize_label(raw_label)
        if class_id is None:
            if raw_label:
                ignored_labels[raw_label] += 1
            continue

        points = shape.get("points") or []
        if len(points) < 3:
            continue

        coords = []
        for point in points:
            if len(point) != 2:
                continue
            point_x, point_y = point
            normalized_x = max(0.0, min(float(point_x), float(image_width))) / max(1.0, float(image_width))
            normalized_y = max(0.0, min(float(point_y), float(image_height))) / max(1.0, float(image_height))
            coords.append(f"{normalized_x:.6f}")
            coords.append(f"{normalized_y:.6f}")

        if len(coords) < 6:
            continue

        yolo_lines.append(f"{class_id} " + " ".join(coords))
        polygon_labels.append(CLASS_NAMES[class_id])

    record = {
        "key": safe_stem(source_root, json_path),
        "json_path": json_path,
        "image_path": image_path,
        "image_suffix": image_path.suffix.lower(),
        "yolo_lines": yolo_lines,
        "polygon_labels": polygon_labels,
        "empty": len(yolo_lines) == 0,
        "source_root": source_root,
    }
    return record, ignored_labels, None


def collect_records(input_roots: list[Path], excluded_dir_names: set[str], skip_empty: bool):
    records = []
    ignored_label_counts = Counter()
    skipped = []

    for source_root in input_roots:
        for json_path in sorted(source_root.rglob("*.json")):
            relative = json_path.relative_to(source_root)
            if should_skip(relative, excluded_dir_names):
                continue

            record, ignored_labels, skip_reason = parse_labelme_json(json_path, source_root)
            ignored_label_counts.update(ignored_labels)
            if skip_reason:
                skipped.append(skip_reason)
                continue
            if record is None:
                continue
            if skip_empty and record["empty"]:
                continue
            records.append(record)

    records.sort(key=lambda record: str(record["json_path"]))
    return records, ignored_label_counts, skipped


def uniform_indices(total: int, target_count: int) -> list[int]:
    if total <= 0 or target_count <= 0:
        return []
    if target_count >= total:
        return list(range(total))
    if target_count == 1:
        return [total // 2]

    step = (total - 1) / (target_count - 1)
    indices = []
    used = set()
    for index in range(target_count):
        candidate = round(index * step)
        if candidate in used:
            continue
        indices.append(candidate)
        used.add(candidate)

    if len(indices) < target_count:
        for candidate in range(total):
            if candidate in used:
                continue
            indices.append(candidate)
            used.add(candidate)
            if len(indices) == target_count:
                break

    return sorted(indices)


def record_class_ids(record: dict[str, Any]) -> set[int]:
    class_ids = set()
    for line in record["yolo_lines"]:
        class_ids.add(int(line.split()[0]))
    return class_ids


def choose_validation_indices(records: list[dict[str, Any]], val_count: int, split_mode: str, seed: int) -> set[int]:
    if split_mode == "random":
        rng = random.Random(seed)
        candidates = list(range(len(records)))
        rng.shuffle(candidates)
        return set(candidates[:val_count])
    return set(uniform_indices(len(records), val_count))


def enforce_train_class_coverage(train_records: list[dict[str, Any]], val_records: list[dict[str, Any]], target_val_count: int):
    all_records = train_records + val_records
    classes_with_data = set().union(*(record_class_ids(record) for record in all_records)) if all_records else set()

    for class_id in sorted(classes_with_data):
        if any(class_id in record_class_ids(record) for record in train_records):
            continue
        for record in list(val_records):
            if class_id not in record_class_ids(record):
                continue
            val_records.remove(record)
            train_records.append(record)
            break

    protected_keys = set()
    class_image_counts = Counter()
    for record in all_records:
        for class_id in record_class_ids(record):
            class_image_counts[class_id] += 1
    singleton_classes = {class_id for class_id, count in class_image_counts.items() if count == 1}
    for record in train_records:
        if record_class_ids(record) & singleton_classes:
            protected_keys.add(record["key"])

    while len(val_records) < target_val_count and len(train_records) > 1:
        candidate = next(
            (record for record in train_records if record["key"] not in protected_keys and not record_class_ids(record)),
            None,
        )
        if candidate is None:
            candidate = next((record for record in train_records if record["key"] not in protected_keys), None)
        if candidate is None:
            break
        train_records.remove(candidate)
        val_records.append(candidate)

    train_records.sort(key=lambda record: str(record["json_path"]))
    val_records.sort(key=lambda record: str(record["json_path"]))


def split_records(records: list[dict[str, Any]], val_ratio: float, split_mode: str, seed: int):
    total = len(records)
    if total == 0:
        return [], []
    if total == 1:
        return records, []

    val_count = max(1, round(total * val_ratio))
    val_count = min(val_count, total - 1)
    val_indices = choose_validation_indices(records, val_count, split_mode, seed)
    train_records = [record for index, record in enumerate(records) if index not in val_indices]
    val_records = [record for index, record in enumerate(records) if index in val_indices]
    enforce_train_class_coverage(train_records, val_records, val_count)
    return train_records, val_records


def prepare_output_dir(output_root: Path, overwrite: bool):
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise SystemExit(f"output directory is not empty, pass --overwrite to replace it: {output_root}")
        shutil.rmtree(output_root)

    for relative in ["images/train", "images/val", "labels/train", "labels/val"]:
        (output_root / relative).mkdir(parents=True, exist_ok=True)


def write_records(records: list[dict[str, Any]], split: str, output_root: Path):
    image_dir = output_root / "images" / split
    label_dir = output_root / "labels" / split
    for record in records:
        image_dst = image_dir / f"{record['key']}{record['image_suffix']}"
        label_dst = label_dir / f"{record['key']}.txt"
        shutil.copy2(record["image_path"], image_dst)
        label_dst.write_text("\n".join(record["yolo_lines"]) + ("\n" if record["yolo_lines"] else ""), encoding="utf-8")


def write_data_yaml(output_root: Path):
    data = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(CLASS_NAMES),
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }
    data_yaml = output_root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return data_yaml


def count_labels(records: list[dict[str, Any]]):
    polygon_counts = Counter()
    image_presence = Counter()
    for record in records:
        labels = record["polygon_labels"]
        polygon_counts.update(labels)
        image_presence.update(set(labels))
    return polygon_counts, image_presence


def write_summary(
    output_root: Path,
    input_roots: list[Path],
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    ignored_label_counts: Counter,
    skipped: list[str],
    args: argparse.Namespace,
):
    all_records = train_records + val_records
    total_polygon_counts, total_image_presence = count_labels(all_records)
    train_polygon_counts, train_image_presence = count_labels(train_records)
    val_polygon_counts, val_image_presence = count_labels(val_records)
    summary = {
        "source_roots": [str(path.resolve()) for path in input_roots],
        "classes": {index: name for index, name in enumerate(CLASS_NAMES)},
        "val_ratio": args.val_ratio,
        "split_mode": args.split_mode,
        "seed": args.seed,
        "images_total": len(all_records),
        "train_images": len(train_records),
        "val_images": len(val_records),
        "empty_images": sum(record["empty"] for record in all_records),
        "class_polygon_counts": dict(total_polygon_counts),
        "class_image_presence": dict(total_image_presence),
        "train_class_polygon_counts": dict(train_polygon_counts),
        "train_class_image_presence": dict(train_image_presence),
        "val_class_polygon_counts": dict(val_polygon_counts),
        "val_class_image_presence": dict(val_image_presence),
        "ignored_label_counts": dict(ignored_label_counts),
        "skipped": skipped,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_path


def run_conversion(args):
    input_roots = [Path(item).resolve() for item in args.inputs]
    output_root = Path(args.output).resolve()
    excluded_dir_names = set(args.exclude_dir or [])

    missing_inputs = [path for path in input_roots if not path.is_dir()]
    if missing_inputs:
        raise SystemExit(f"missing input directories: {missing_inputs}")

    records, ignored_label_counts, skipped = collect_records(
        input_roots=input_roots,
        excluded_dir_names=excluded_dir_names,
        skip_empty=args.skip_empty,
    )
    if not records:
        raise SystemExit("no Labelme records found after filtering")

    train_records, val_records = split_records(records, args.val_ratio, args.split_mode, args.seed)
    prepare_output_dir(output_root, args.overwrite)
    write_records(train_records, "train", output_root)
    write_records(val_records, "val", output_root)
    data_yaml = write_data_yaml(output_root)
    summary_path = write_summary(output_root, input_roots, train_records, val_records, ignored_label_counts, skipped, args)

    print(f"Converted dataset: {output_root}")
    print(f"Train images:      {len(train_records)}")
    print(f"Val images:        {len(val_records)}")
    print(f"Data yaml:         {data_yaml}")
    print(f"Summary:           {summary_path}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_conversion(args)


if __name__ == "__main__":
    main()
