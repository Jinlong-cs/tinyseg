# Data Reproducibility

TinySeg expects raw annotated data to stay outside the repository. This repository stores the conversion code and small dataset recipes needed to rebuild YOLO segmentation datasets from that raw data.

## Raw Data Contract

Prepare a raw data directory that contains images and Labelme JSON files side by side:

```text
<raw-root>/
└── day/
    ├── office/
    ├── office_test/
    └── park/
        └── .../
            └── infra1/
                ├── 000001.jpg
                ├── 000001.json
                ├── 000002.jpg
                └── 000002.json
```

The converter scans JSON files recursively. For each JSON file it finds the image from `imagePath` first, then falls back to the same stem with common image extensions.

## Labels

The current reproducible recipe is a 2-class segmentation task:

| Class id | Name | Labelme aliases kept |
| ---: | --- | --- |
| `0` | `drivable` | `drivable`, `drivable_area`, `free_traversable`, `cautious_traversable`, `roadway_nonped`, `ground`, `floor`, `地面` |
| `1` | `stairs` | `stairs`, `stair`, `stairs_escalator`, `楼梯` |

All other labels are ignored and become background. Ignored label counts are written to `summary.json`.

## Config-Driven Conversion

Use a committed recipe so another machine can reproduce the same split and filtering rules:

```bash
uv run python prepare_yolo_dataset.py \
    --config configs/datasets/drivable_stairs_discover_day_infra1.yaml \
    --raw-root /path/to/dataset_discover \
    --overwrite
```

The example recipe:

- reads `day/office`, `day/office_test`, and `day/park` under `--raw-root`
- keeps only paths matching `infra1`
- keeps only `drivable` and `stairs`
- drops images with no kept polygons
- uses temporal split with `val_ratio=0.10`
- writes a portable `data.yaml` with `path: .`

To build another infra dataset, copy the recipe and change:

```yaml
name: drivable_stairs_discover_day_infra3_v1
output: data/drivable_stairs_discover_day_infra3_v1
scan:
  include_path_regex:
    - '(^|/)infra3(/|$)'
```

## Direct CLI Conversion

For one-off conversion without a recipe:

```bash
uv run python convert_labelme_drivable_stairs.py \
    --inputs /path/to/dataset_discover/day/office \
             /path/to/dataset_discover/day/office_test \
             /path/to/dataset_discover/day/park \
    --output data/drivable_stairs_discover_day_infra1_v1 \
    --include-path-regex '(^|/)infra1(/|$)' \
    --val-ratio 0.10 \
    --split-mode temporal \
    --skip-empty \
    --yaml-path-mode relative \
    --overwrite
```

Use the config-driven command for experiments that should be reproduced later.

## Output Files

The converter writes a standard YOLO-seg dataset:

```text
data/<dataset-name>/
├── data.yaml
├── summary.json
├── conversion_manifest.yaml
├── source_index.jsonl
├── splits/
│   ├── train.txt
│   └── val.txt
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Important files:

- `data.yaml`: training entry for Ultralytics.
- `summary.json`: counts, split settings, ignored labels, and skipped files.
- `conversion_manifest.yaml`: exact recipe metadata and conversion arguments.
- `source_index.jsonl`: one line per converted image with source JSON/image and output paths.
- `splits/train.txt` and `splits/val.txt`: output image membership for audit.

## Reproducibility Checklist

Before training, check:

- `summary.json` has the expected `images_total`, `train_images`, and `val_images`.
- `class_image_presence` includes the classes expected for the task.
- `ignored_label_counts` only contains labels that should become background.
- `source_index.jsonl` points to the intended infra folders.
- `data.yaml` can be passed directly to `train_yolov26.py`.
