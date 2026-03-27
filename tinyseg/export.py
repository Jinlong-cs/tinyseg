import argparse
import shutil
from pathlib import Path

RDK_OUTPUT_NAMES = [
    "cls_s8",
    "box_s8",
    "mc_s8",
    "cls_s16",
    "box_s16",
    "mc_s16",
    "cls_s32",
    "box_s32",
    "mc_s32",
    "proto",
]


def build_parser():
    parser = argparse.ArgumentParser(description="Export an RDK X5 friendly ONNX from a YOLO segmentation checkpoint.")
    parser.add_argument("--pt", required=True, help="Path to a trained .pt checkpoint.")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset.")
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=[640],
        help="Export image size. Use one value for square input or two values for h w.",
    )
    parser.add_argument("--output", help="Optional output ONNX path.")
    return parser


def normalize_imgsz(imgsz_values):
    if len(imgsz_values) == 1:
        return imgsz_values[0]
    if len(imgsz_values) == 2:
        return tuple(imgsz_values)
    raise SystemExit("--imgsz expects one value or two values: h w")


def output_path_for(pt_path, output, imgsz):
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    pt_path = Path(pt_path)
    if isinstance(imgsz, int):
        stem = f"{pt_path.stem}_{imgsz}x{imgsz}.onnx"
    else:
        stem = f"{pt_path.stem}_{imgsz[0]}x{imgsz[1]}.onnx"
    return pt_path.with_name(stem)


def export_checkpoint_fallback(pt_path, opset=11, imgsz=640, output=None):
    import torch
    from ultralytics import YOLO
    from tinyseg.ultralytics_rdk import patch_model_for_rdk

    model = YOLO(pt_path)
    patch_model_for_rdk(model.model.model)
    model.model.eval()

    if isinstance(imgsz, int):
        height = width = imgsz
    else:
        height, width = imgsz

    output_path = output_path_for(pt_path, output, imgsz)
    example = torch.randn(1, 3, height, width, dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            model.model,
            example,
            str(output_path),
            input_names=["images"],
            output_names=RDK_OUTPUT_NAMES,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )

    return output_path.resolve()


def export_checkpoint(pt_path, opset=11, imgsz=640, output=None):
    from ultralytics import YOLO
    from tinyseg.ultralytics_rdk import patch_model_for_rdk

    model = YOLO(pt_path)
    patch_model_for_rdk(model.model.model)
    try:
        exported_path = Path(model.export(format="onnx", simplify=False, opset=opset, imgsz=imgsz))
        if output:
            output_path = output_path_for(pt_path, output, imgsz)
            shutil.copy2(exported_path, output_path)
            return output_path.resolve()
        return exported_path.resolve()
    except Exception as exc:
        print(f"Ultralytics export failed, falling back to torch.onnx.export: {exc}")
        return export_checkpoint_fallback(pt_path=pt_path, opset=opset, imgsz=imgsz, output=output)


def main(argv=None):
    args = build_parser().parse_args(argv)
    exported_path = export_checkpoint(
        pt_path=args.pt,
        opset=args.opset,
        imgsz=normalize_imgsz(args.imgsz),
        output=args.output,
    )
    print(f"Exported ONNX: {exported_path}")


if __name__ == "__main__":
    main()
