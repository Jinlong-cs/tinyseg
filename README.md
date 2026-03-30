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

uv run python merge_yolo_datasets.py --help
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
├── merge_yolo_datasets.py
├── train_yolov26.py
├── export_onnx.py
├── quantize_x5.py
├── verify_board.py
├── data/
├── experiments/
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
    ├── merge.py
    ├── quantize.py
    ├── rdk_x5_config.py
    ├── train.py
    ├── ultralytics_rdk.py
    └── verify.py
```

Repository storage convention:
- `data/`: converted YOLO-format datasets
- `experiments/`: curated dated artifacts worth keeping in git
- `runs/`: Ultralytics training outputs
- `outputs/`: ONNX, compiled models, board verification outputs

Only curated experiment artifacts should be committed. Temporary datasets, checkpoints, and one-off outputs should stay untracked.

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
    --model yolo26n-seg.pt \
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

## Experiments

Use this section as an append-only experiment log:
- add one summary row per dated experiment
- keep one dated subsection with scenes, metrics, and board-side outputs
- only link artifacts that are worth preserving in git

| Date | Experiment | Scenes | Dataset | Validation Summary | Board Summary | Artifacts |
| --- | --- | --- | --- | --- | --- | --- |
| `2026-03-26` | `office_test_manualclean_v2` | `Office_test` | `652 images` | `9-class deployment comparison` | `yolov26_9cls: 14.909 ms, fg mIoU 0.7455`; `yolov26_9cls_dfl_adapter: 29.686 ms, fg mIoU 0.7347` | `Not kept in repo` |
| `2026-03-27` | `open9_corridor_elevator_office_v1` | `corridor + elevator + Office_test` | `824 images (700 train / 124 val)` | `best mask mAP50 0.863, best mask mAP50-95 0.621` | `fg mIoU 0.7465, pixel acc 0.9334, latency unavailable in report` | [video](experiments/2026-03-27_open9_corridor_elevator_office_v1/mixed_val_previews.mp4), [pt](experiments/2026-03-27_open9_corridor_elevator_office_v1/best.pt), [onnx](experiments/2026-03-27_open9_corridor_elevator_office_v1/open9_corridor_elevator_office_v1_best_352x640.onnx), [bin](experiments/2026-03-27_open9_corridor_elevator_office_v1/open9_corridor_elevator_office_v1_best_352x640_bayese_640x352_nv12.bin), [wandb](https://wandb.ai/eddie18361268318-discover/tinyseg/runs/z3fth4j9) |

### 2026-03-27: Open9 Corridor + Elevator + Office

Scenes:
- `corridor`
- `elevator`
- `Office_test`

Dataset:
- `corridor_yolo_1`: `74` images
- `elevator_yolo`: `98` images
- `Office_test_yolo`: `652` images
- merged total: `824` images
- train / val split: `700 / 124`

Training:
- W&B run: [open9_corridor_elevator_office_v1_20260327](https://wandb.ai/eddie18361268318-discover/tinyseg/runs/z3fth4j9)

| Metric Group | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| Box | `0.886` | `0.829` | `0.907` | `0.750` |
| Mask | `0.842` | `0.811` | `0.863` | `0.621` |

Board-side result:
- board: `Horizon X5`
- eval split: merged validation set, `124` images
- foreground mIoU: `0.7465`
- pixel accuracy: `0.9334`
- latency: unavailable in this run because the board report stored `NaN` for all `/segment/stats` rows

Artifacts kept in repo:
- [best.pt](experiments/2026-03-27_open9_corridor_elevator_office_v1/best.pt)
- [open9_corridor_elevator_office_v1_best_352x640.onnx](experiments/2026-03-27_open9_corridor_elevator_office_v1/open9_corridor_elevator_office_v1_best_352x640.onnx)
- [open9_corridor_elevator_office_v1_best_352x640_bayese_640x352_nv12.bin](experiments/2026-03-27_open9_corridor_elevator_office_v1/open9_corridor_elevator_office_v1_best_352x640_bayese_640x352_nv12.bin)
- [mixed_val_previews.mp4](experiments/2026-03-27_open9_corridor_elevator_office_v1/mixed_val_previews.mp4)

Board preview video:
experiments/2026-03-27_open9_corridor_elevator_office_v1/mixed_val_previews.mp4

Fallback link:
- [mixed_val_previews.mp4](experiments/2026-03-27_open9_corridor_elevator_office_v1/mixed_val_previews.mp4)
