from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
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


def letterbox(image_bgr, target_hw, color=(114, 114, 114)):
    target_h, target_w = target_hw
    src_h, src_w = image_bgr.shape[:2]
    scale = min(target_w / src_w, target_h / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))

    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), color, dtype=np.uint8)

    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas


def preprocess_for_calibration(image_bgr, target_hw, preprocess="letterbox"):
    if preprocess == "letterbox":
        prepared = letterbox(image_bgr, target_hw)
    elif preprocess == "resize":
        target_h, target_w = target_hw
        prepared = cv2.resize(image_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    else:
        raise SystemExit(f"Unsupported preprocess mode: {preprocess}")

    rgb = cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb, (2, 0, 1))
    nchw = np.expand_dims(chw, axis=0).astype(np.float32)
    return nchw


def write_calibration_tensors(image_paths, output_dir, target_hw, preprocess="letterbox"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written_files = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        tensor = preprocess_for_calibration(image, target_hw, preprocess=preprocess)
        output_path = output_dir / f"{image_path.name}.rgbchw"
        tensor.tofile(output_path)
        written_files.append(output_path)

    return written_files
