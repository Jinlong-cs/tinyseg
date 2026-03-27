# TinySeg

TinySeg is a small training-side repository for YOLO segmentation on Horizon RDK X5.

It keeps the model workflow in one place:
- train a segmentation model
- export a board-friendly ONNX
- run PTQ quantization and compile a `.bin`
- verify the compiled model on the board

It intentionally does not include:
- annotation tools
- dataset generation
- pseudo-label pipelines
- ROS runtime deployment code

## Overview

The repository is organized around a simple rule:
- thin top-level entry scripts
- reusable logic inside the `tinyseg/` package
- explicit inputs and outputs for every stage

The result is easier to read, easier to copy into a new workspace, and easier to maintain as a standalone product.

## Quick Start

```bash
git clone git@github.com:Jinlong-cs/tinyseg.git
cd tinyseg

uv venv
uv sync

uv run python train_yolov26.py --help
uv run python export_onnx.py --help
uv run python quantize_x5.py --help
uv run python verify_board.py --help
```

## Project Structure

```text
tinyseg/
├── README.md
├── pyproject.toml
├── train_yolov26.py
├── export_onnx.py
├── quantize_x5.py
├── verify_board.py
├── dev/
│   ├── Dockerfile
│   ├── build.sh
│   └── run.sh
└── tinyseg/
    ├── __init__.py
    ├── calibration.py
    ├── export.py
    ├── quantize.py
    ├── rdk_x5_config.py
    ├── train.py
    ├── ultralytics_rdk.py
    └── verify.py
```

## Training

```bash
uv run python train_yolov26.py \
    --data data/office_manualclean/data.yaml \
    --model yolo11n-seg.pt \
    --epochs 150 \
    --imgsz 640 \
    --batch 8 \
    --device 0 \
    --name office_manualclean
```

Ultralytics outputs follow the standard layout under `runs/seg/<name>/`.

## Export ONNX

```bash
uv run python export_onnx.py \
    --pt runs/seg/office_manualclean/weights/best.pt \
    --imgsz 352 640 \
    --output artifacts/office_manualclean/best_352x640.onnx
```

The export step patches the Ultralytics model into an RDK-friendly output form before ONNX conversion.

## Quantize And Compile

```bash
uv run python quantize_x5.py \
    --workspace . \
    --onnx artifacts/office_manualclean/best_352x640.onnx \
    --data-yaml data/office_manualclean/data.yaml \
    --cal-split train \
    --output-dir artifacts/office_manualclean/rdk_x5 \
    --preprocess letterbox
```

This stage:
- samples calibration images
- writes `config.yaml`
- launches `hb_mapper` in Docker
- saves the final `.bin` next to the config and logs

## Board Verification

```bash
uv run python verify_board.py \
    --host 192.168.31.63 \
    --user sunrise \
    --password sunrise \
    --model-file artifacts/office_manualclean/rdk_x5/best_352x640_bayese_640x352_nv12.bin \
    --input-bin sample.rgbchw \
    --output-dir artifacts/office_manualclean/board_verify
```

The verification script uploads the compiled model and one prepared input tensor, runs `hrt_model_exec infer`, and downloads the dump files for inspection.
