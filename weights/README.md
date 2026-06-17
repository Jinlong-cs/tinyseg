# TinySeg Pretrained Weights

These checkpoints are intended as starting weights for future TinySeg
fine-tuning on new DiscoverSegment data. They are Ultralytics YOLO segmentation
`.pt` checkpoints.

## 2026-06-14 Villa Models

Dataset source:
`backblaze:vastai-yzf/segment_data/DiscoverSegment` after merging
`Villa-1---ok` and `villa-2---ok`.

Training code:
`Jinlong-cs/tinyseg` main at
`025bf40651736a354bf07843cb3423e1d883c5e6`.

Training settings:
YOLO26n segmentation, image size `640`, epochs `200`, patience `30`, W&B
online. These runs used batch `16`; current repo defaults may differ.

| File | Split | Train / Val | Best mask mAP50-95 | SHA256 |
| --- | --- | ---: | ---: | --- |
| `tinyseg_villa_0614_infra1_yolo26n_best.pt` | infra1 | `7167 / 800` | `0.85258` at epoch `182` | `ed36ebb1874ef936f95f8b1cef50f53c01ee999534611caadefc4ee0da9aaeb8` |
| `tinyseg_villa_0614_infra34_yolo26n_best.pt` | infra3+infra4 | `9211 / 1025` | `0.86971` at epoch `187` | `04dd2baee4f7a86676a8bc6ac27083d8731b9f0d962a0208e3f6ab419ffd7ed3` |

Use the matching checkpoint for the camera split you are fine-tuning:

```bash
uv run python train_yolov26.py \
    --train-list data/DiscoverSegment/infra1_train_list.txt \
    --val-list data/DiscoverSegment/infra1_val_list.txt \
    --model weights/tinyseg_villa_0614_infra1_yolo26n_best.pt
```

For side cameras, use `tinyseg_villa_0614_infra34_yolo26n_best.pt` and the
infra34 split. Do not mix infra1 into infra34 unless that is an explicit
experiment.
