# TinySeg X5 Image Deploy

This folder contains a standalone C++ image inference tool for Horizon RDK X5.

It reuses the same `SegmentInfer` runtime as the 2-class TinyNav deployment, but runs on a directory of images instead of ROS topics.

## Build On X5

```bash
cd tinyseg/tinyseg/deploy/cpp
rm -rf build
cmake -S . -B build
cmake --build build -j1
```

## Run On X5

```bash
./build/segment_image_infer \
  --model /path/to/tinyseg_2classes_infra1_352x640_bayese_640x352_nv12.bin \
  --input-dir /path/to/images \
  --output-dir /path/to/output
```

The runtime reads the input tensor shape from the compiled model. For the current TinySeg deployment model, export ONNX with `--imgsz 352 640` and compile the resulting `640x352_nv12` model; the default preprocessing letterboxes each source image into that shape using the same `114` padding value as YOLO/Ultralytics calibration.

Outputs:
- `overlay/*.png`: source image with color overlay
- `mask_color/*.png`: rendered class mask
- `summary.json`: per-image latency and output paths
