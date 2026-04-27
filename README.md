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
- `runs/`: Ultralytics training outputs
- `outputs/`: ONNX, compiled models, board verification outputs

Temporary datasets, checkpoints, exported models, and one-off outputs should stay untracked unless explicitly needed for a release.

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

W&B logging is enabled by default. Provide `WANDB_API_KEY`, pass `--wandb-api-key`, or create a local `.wandb_api_key` file before training. Each run logs startup train GT visualizations under `train/ground_truth_samples` and end-of-training validation GT-vs-prediction comparisons under `val/prediction_samples`; prediction samples default to the deployment input size `352 640` and can be changed with `--wandb-pred-imgsz`. Use `--no-wandb`, `--wandb-sample-count`, or `--wandb-pred-sample-count` to control logging.

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

The C++ runtime reads the model input shape directly from the compiled `.bin`; with TinySeg's `--imgsz 352 640` export and `640x352_nv12` compile, preprocessing letterboxes into the deployment input size using YOLO-compatible `114` padding.

Outputs:
- `overlay/*.png`: source image with segmentation overlay
- `mask_color/*.png`: rendered class mask
- `summary.json`: per-image latency and saved paths

## Experiment Tracking

Experiment artifacts are no longer committed under `experiments/`. Keep training history and sample visualizations in W&B, and keep local generated files under `runs/` or `outputs/`.

For reproducible runs, record:
- dataset recipe or data list files
- training command and model checkpoint source
- W&B run URL
- exported ONNX or compiled model location outside git
