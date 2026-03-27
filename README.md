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
├── data/
├── runs/
├── outputs/
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

Repository storage convention:
- `data/`: converted YOLO-format datasets
- `runs/`: Ultralytics training outputs
- `outputs/`: ONNX, compiled models, board verification outputs

Only the folder skeleton is kept in git. Generated datasets, checkpoints, and experiment artifacts should stay untracked.

## Dataset Labels

TinySeg currently uses the following 9 segmentation labels:

| Label | Meaning |
| --- | --- |
| `free_traversable` | Normal traversable area |
| `cautious_traversable` | Traversable but should slow down |
| `stairs_escalator` | Stairs or escalator, not traversable |
| `dropoff_edge` | High-risk edge such as curb or step boundary |
| `roadway_nonped` | Roadway or mixed traffic area that should not be entered by default |
| `fixed_barrier` | Fixed obstacle such as wall, cabinet, pole, or railing |
| `glass_barrier` | Transparent obstacle such as glass door or glass wall |
| `person` | Person or crowd that should be avoided with high priority |
| `movable_obstacle` | Movable obstacle such as cart, wheelchair, box, or bike |

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
    --output outputs/office_manualclean/best_352x640.onnx
```

The export step patches the Ultralytics model into an RDK-friendly output form before ONNX conversion.

## Quantize And Compile

```bash
uv run python quantize_x5.py \
    --workspace . \
    --onnx outputs/office_manualclean/best_352x640.onnx \
    --data-yaml data/office_manualclean/data.yaml \
    --cal-split train \
    --output-dir outputs/office_manualclean/rdk_x5 \
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
    --model-file outputs/office_manualclean/rdk_x5/best_352x640_bayese_640x352_nv12.bin \
    --input-bin sample.rgbchw \
    --output-dir outputs/office_manualclean/board_verify
```

The verification script uploads the compiled model and one prepared input tensor, runs `hrt_model_exec infer`, and downloads the dump files for inspection.

## X5 Inference Speed

Latest board-side timing snapshot on Horizon X5:

- dataset: `Office_test manualclean_v2`
- images: `652`
- input size: `352 x 640`
- date: `2026-03-26`

| Route | Preprocess Mean | Infer Mean | Postprocess Mean | Total Mean | Total P95 | Foreground mIoU | Pixel Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `yolov26_9cls` | 1.060 ms | 7.637 ms | 6.211 ms | 14.909 ms | 18.069 ms | 0.7455 | 0.9443 |
| `yolov26_9cls_dfl_adapter` | 1.058 ms | 21.386 ms | 7.240 ms | 29.686 ms | 38.407 ms | 0.7347 | 0.9462 |
