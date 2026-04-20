from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


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
