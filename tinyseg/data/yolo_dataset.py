from __future__ import annotations

import random
from pathlib import Path

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_images_from_dir(image_dir):
    image_dir = Path(image_dir)
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)


def read_split_sources(data_yaml_path, split):
    data_yaml_path = Path(data_yaml_path).resolve()
    data = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))

    root = Path(data.get("path", data_yaml_path.parent))
    if not root.is_absolute():
        root = (data_yaml_path.parent / root).resolve()

    split_value = data[split]
    if isinstance(split_value, str):
        split_sources = [split_value]
    else:
        split_sources = list(split_value)

    resolved_sources = []
    for source in split_sources:
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = (root / source_path).resolve()
        resolved_sources.append(source_path)
    return resolved_sources


def collect_images_from_split(data_yaml_path, split="train"):
    images = []
    for source in read_split_sources(data_yaml_path, split):
        if source.is_dir():
            images.extend(collect_images_from_dir(source))
            continue

        if source.is_file() and source.suffix.lower() == ".txt":
            for raw_line in source.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                line_path = Path(line)
                if not line_path.is_absolute():
                    line_path = (source.parent / line_path).resolve()
                images.append(line_path)
            continue

        raise SystemExit(f"Unsupported split source: {source}")

    seen = set()
    unique_images = []
    for image_path in images:
        if image_path in seen:
            continue
        seen.add(image_path)
        unique_images.append(image_path)
    return unique_images


def sample_images(images, sample_num, seed=42):
    images = list(images)
    if sample_num <= 0 or sample_num >= len(images):
        return images

    rng = random.Random(seed)
    sampled = rng.sample(images, sample_num)
    return sorted(sampled)
