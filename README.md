# TinySeg

TinySeg is a small training-side repository for YOLO segmentation on Horizon RDK X5.

It keeps the model workflow in one place:
- convert Labelme annotations to a 2-class YOLO-seg dataset
- rebuild YOLO training datasets from committed conversion recipes
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
- reusable logic grouped by stage inside the `tinyseg/` package
- explicit inputs and outputs for every stage

The result is easier to read, easier to copy into a new workspace, and easier to maintain as a standalone product.

## Quick Start

```bash
git clone git@github.com:Jinlong-cs/tinyseg.git
cd tinyseg

uv venv
uv sync

uv run python prepare_yolo_dataset.py --help
uv run python convert_labelme_drivable_stairs.py --help
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
├── prepare_yolo_dataset.py
├── convert_labelme_drivable_stairs.py
├── train_yolov26.py
├── export_onnx.py
├── quantize_x5.py
├── verify_board.py
├── data/
├── experiments/
├── runs/
├── outputs/
├── configs/
│   └── datasets/
├── docs/
│   └── data_reproducibility.md
├── dev/
│   ├── Dockerfile
│   ├── build.sh
│   └── run.sh
└── tinyseg/
    ├── __init__.py
    ├── data/
    │   ├── prepare_yolo_dataset.py
    │   ├── labelme_drivable_stairs.py
    │   └── yolo_dataset.py
    ├── training/
    │   ├── train.py
    │   └── wandb_logger.py
    ├── export_onnx/
    │   ├── export.py
    │   └── ultralytics_rdk.py
    ├── quantization/
    │   ├── calibration.py
    │   ├── quantize.py
    │   └── rdk_x5_config.py
    └── deploy/
        ├── verify.py
        └── cpp/
            ├── CMakeLists.txt
            ├── segment_image_infer.cpp
            ├── segment_infer.cpp
            └── segment_infer.h
```

Repository storage convention:
- `data/`: converted YOLO-format datasets
- `experiments/`: curated dated artifacts worth keeping in git
- `runs/`: Ultralytics training outputs
- `outputs/`: ONNX, compiled models, board verification outputs

Only curated experiment artifacts should be committed. Temporary datasets, checkpoints, and one-off outputs should stay untracked.

## Dataset Labels

TinySeg is configured for two foreground segmentation labels:

| Label | Meaning |
| --- | --- |
| `drivable` | Traversable floor or ground region |
| `stairs` | Stairs or stair-like area |

The Labelme converter keeps only labels that map to these two classes. Labels such as `dangerous_area`, `sky`, `car`, `person`, or `dropoff` are ignored and become background.

## Convert Labelme Dataset

For reproducible experiments, prefer the recipe-driven converter. The raw data stays outside this repo; the recipe records input subfolders, path filters, split settings, and output location.

```bash
uv run python prepare_yolo_dataset.py \
    --config configs/datasets/drivable_stairs_discover_day_infra1.yaml \
    --raw-root /path/to/dataset_discover \
    --overwrite
```

This writes a YOLO-seg dataset plus audit files:
- `data.yaml`: Ultralytics training entry
- `summary.json`: counts, ignored labels, split settings
- `conversion_manifest.yaml`: exact recipe metadata and conversion arguments
- `source_index.jsonl`: source JSON/image to output image/label mapping
- `splits/train.txt` and `splits/val.txt`: split membership

See [data reproducibility docs](docs/data_reproducibility.md) for raw data layout, label aliases, and recipe guidelines.

For ad-hoc conversion, call the lower-level Labelme converter directly:

```bash
uv run python convert_labelme_drivable_stairs.py \
    --inputs /path/to/dataset_discover \
    --output data/drivable_stairs_discover_v1 \
    --val-ratio 0.10 \
    --split-mode temporal \
    --yaml-path-mode relative \
    --overwrite
```

The converter recursively scans Labelme JSON files, writes mirrored YOLO-seg `images/...` and `labels/...`, and creates:
- `train_list.txt`
- `val_list.txt`

Each list file contains image paths relative to the converted dataset root. The split is applied per source leaf directory so `day_office/infra1`, `day_park/infra1`, and similar groups all keep their own train/val coverage.

## Training

```bash
uv run python train_yolov26.py \
    --train-list data/drivable_stairs_discover_v1/train_list.txt \
    --val-list data/drivable_stairs_discover_v1/val_list.txt \
    --class-names drivable,stairs \
    --model yolo26n-seg.pt \
    --epochs 150 \
    --imgsz 640 \
    --batch 8 \
    --device 0 \
    --name drivable_stairs_discover_v1
```

W&B logging is enabled by default. Provide `WANDB_API_KEY`, pass `--wandb-api-key`, or create a local `.wandb_api_key` file before training. Each run logs startup train-sample visualizations under `train/samples`; use `--no-wandb` to disable logging or `--wandb-sample-count` to change how many labeled samples are uploaded.

Ultralytics outputs follow the standard layout under `runs/seg/<name>/`.

## Export ONNX

```bash
uv run python export_onnx.py \
    --pt runs/seg/drivable_stairs_discover_v1/weights/best.pt \
    --imgsz 352 640 \
    --output outputs/drivable_stairs_discover_v1/best_352x640.onnx
```

The export step patches the Ultralytics model into an RDK-friendly output form before ONNX conversion.

## Quantize And Compile

```bash
uv run python quantize_x5.py \
    --workspace . \
    --onnx outputs/drivable_stairs_discover_v1/best_352x640.onnx \
    --data-yaml data/drivable_stairs_discover_v1/data.yaml \
    --cal-split train \
    --output-dir outputs/drivable_stairs_discover_v1/rdk_x5 \
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
    --model-file outputs/drivable_stairs_discover_v1/rdk_x5/best_352x640_bayese_640x352_nv12.bin \
    --input-bin sample.rgbchw \
    --output-dir outputs/drivable_stairs_discover_v1/board_verify
```

The verification script uploads the compiled model and one prepared input tensor, runs `hrt_model_exec infer`, and downloads the dump files for inspection.

## X5 Image Inference

For board-side visualization on a directory of images, use the standalone C++ tool under `tinyseg/deploy/cpp`.

Build on the X5 board:

```bash
cd tinyseg/tinyseg/deploy/cpp
rm -rf build
cmake -S . -B build
cmake --build build -j1
```

Run on the X5 board:

```bash
./build/segment_image_infer \
    --model /path/to/model.bin \
    --input-dir /path/to/images \
    --output-dir /path/to/output
```

Outputs:
- `overlay/*.png`: source image with segmentation overlay
- `mask_color/*.png`: rendered class mask
- `summary.json`: per-image latency and saved paths

## Experiments

Use this section as an append-only experiment log:
- add one summary row per dated experiment
- keep one dated subsection with scenes, metrics, and board-side outputs
- only link artifacts that are worth preserving in git

| Date | Experiment | Scenes | Dataset | Validation Summary | Board Summary | Artifacts |
| --- | --- | --- | --- | --- | --- | --- |
| `2026-04-20` | `drivable_stairs_discover_day_infra1_v1` | `office + office_test + park`, `infra1 only` | `522 images (470 train / 52 val)` | `2-class best.pt: mask mAP50 0.995, mask mAP50-95 0.974` | `RDK X5 bag replay smoke test passed; 8 overlay frames saved from infra1 rosbag` | [pt](experiments/2026-04-20_drivable_stairs_discover_day_infra1_v1/best.pt), [onnx](experiments/2026-04-20_drivable_stairs_discover_day_infra1_v1/drivable_stairs_discover_day_infra1_v1_best_352x640.onnx), [bin](experiments/2026-04-20_drivable_stairs_discover_day_infra1_v1/drivable_stairs_discover_day_infra1_v1_best_352x640_bayese_640x352_nv12.bin) |
| `2026-03-26` | `office_test_manualclean_v2` | `Office_test` | `652 images` | `9-class deployment comparison` | `yolov26_9cls: 14.909 ms, fg mIoU 0.7455`; `yolov26_9cls_dfl_adapter: 29.686 ms, fg mIoU 0.7347` | `Not kept in repo` |
| `2026-03-27` | `open9_corridor_elevator_office_v1` | `corridor + elevator + Office_test` | `824 images (700 train / 124 val)` | `best mask mAP50 0.863, best mask mAP50-95 0.621` | `fg mIoU 0.7465, pixel acc 0.9334, latency unavailable in report` | [gif](experiments/2026-03-27_open9_corridor_elevator_office_v1/mixed_val_previews.gif), [pt](experiments/2026-03-27_open9_corridor_elevator_office_v1/best.pt), [onnx](experiments/2026-03-27_open9_corridor_elevator_office_v1/open9_corridor_elevator_office_v1_best_352x640.onnx), [bin](experiments/2026-03-27_open9_corridor_elevator_office_v1/open9_corridor_elevator_office_v1_best_352x640_bayese_640x352_nv12.bin), [wandb](https://wandb.ai/eddie18361268318-discover/tinyseg/runs/z3fth4j9) |

### 2026-04-20: Drivable + Stairs Discover Day Infra1

Scenes:
- `office`
- `office_test`
- `park`
- `infra1 only`

Dataset:
- total: `522` images
- train / val split: `470 / 52`
- task: `2-class segmentation`
- labels kept: `drivable`, `stairs`

Training:
- input training size: `imgsz=640`
- deployment export size: `352x640`
- run dir: `/home/supernova/wujinlong/labelme-with-segment-anything/data_loop/runs/segment/runs/seg/drivable_stairs_discover_day_infra1_v1_20260420`

Validation on `best.pt`:

| Scope | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| Box all | `0.996` | `0.989` | `0.995` | `0.958` |
| Mask all | `0.996` | `0.989` | `0.995` | `0.974` |
| Mask drivable | `1.000` | `0.978` | `0.995` | `0.989` |
| Mask stairs | `0.992` | `1.000` | `0.995` | `0.959` |

Board-side result:
- board: `Horizon RDK X5`
- input bag: `rosbag2_2026_03_13-16_25_22`
- replay topic: `/camera/camera/infra1/image_rect_raw`
- deployed model: `2-class .bin`
- smoke result: model loaded and overlay visualization published correctly
- saved preview: `8` board-side visualization frames
- CPU frequency lock: `performance @ 1500000 kHz`

Artifacts kept in repo:
- [best.pt](experiments/2026-04-20_drivable_stairs_discover_day_infra1_v1/best.pt)
- [drivable_stairs_discover_day_infra1_v1_best_352x640.onnx](experiments/2026-04-20_drivable_stairs_discover_day_infra1_v1/drivable_stairs_discover_day_infra1_v1_best_352x640.onnx)
- [drivable_stairs_discover_day_infra1_v1_best_352x640_bayese_640x352_nv12.bin](experiments/2026-04-20_drivable_stairs_discover_day_infra1_v1/drivable_stairs_discover_day_infra1_v1_best_352x640_bayese_640x352_nv12.bin)

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
- [mixed_val_previews.gif](experiments/2026-03-27_open9_corridor_elevator_office_v1/mixed_val_previews.gif)

Board preview video:

![2026-03-27 board preview](experiments/2026-03-27_open9_corridor_elevator_office_v1/mixed_val_previews.gif)
