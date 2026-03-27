import argparse
import shutil
import subprocess
from pathlib import Path


DEFAULT_DOCKER_IMAGE = "registry.d-robotics.cc/deliver/hub.hobot.cc/aitools/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8"


def build_parser():
    parser = argparse.ArgumentParser(description="Run RDK X5 PTQ quantization and compile a .bin model.")
    parser.add_argument("--onnx", required=True, help="Path to input ONNX model.")
    parser.add_argument("--cal-images", help="Calibration image directory.")
    parser.add_argument("--data-yaml", help="YOLO data.yaml used to collect calibration images from a dataset split.")
    parser.add_argument("--cal-split", default="train", help="Dataset split used with --data-yaml, usually train or val.")
    parser.add_argument("--output-dir", required=True, help="Directory for mapper outputs.")
    parser.add_argument("--workspace", default=".", help="Workspace mounted into Docker.")
    parser.add_argument("--docker-image", default=DEFAULT_DOCKER_IMAGE, help="Docker image for the Horizon toolchain.")
    parser.add_argument("--cal-sample-num", type=int, default=50, help="Calibration sample count.")
    parser.add_argument("--cal-seed", type=int, default=42, help="Sampling seed for calibration image selection.")
    parser.add_argument(
        "--preprocess",
        choices=["letterbox", "resize"],
        default="letterbox",
        help="Calibration preprocessing mode. letterbox is recommended to stay aligned with YOLO-style preprocessing.",
    )
    parser.add_argument("--jobs", type=int, default=8, help="Parallel jobs for mapper/compile.")
    parser.add_argument("--optimize-level", default="O3", help="Mapper optimize level.")
    parser.add_argument("--quantized", choices=["int8", "int16"], default="int8", help="Quantization target.")
    parser.add_argument(
        "--keep-workspace",
        action="store_true",
        help="Keep generated calibration tensors and intermediate workspace under output-dir.",
    )
    parser.add_argument(
        "--mapper-script",
        help="Deprecated compatibility argument. It is ignored because TinySeg now manages config and calibration data internally.",
    )
    return parser


def to_workspace_relative(path_value, workspace):
    path = Path(path_value)
    if not path.is_absolute():
        return path

    resolved_path = path.resolve()
    resolved_workspace = workspace.resolve()
    try:
        return resolved_path.relative_to(resolved_workspace)
    except ValueError as exc:
        raise SystemExit(f"path must stay under workspace: {resolved_path}") from exc


def get_onnx_input_hw(onnx_path):
    import onnx

    model = onnx.load(Path(onnx_path))
    dims = model.graph.input[0].type.tensor_type.shape.dim
    height = dims[2].dim_value
    width = dims[3].dim_value
    if not height or not width:
        raise SystemExit(f"ONNX input shape must be static NCHW, got height={height}, width={width}")
    return int(height), int(width)


def collect_calibration_images(args):
    from tinyseg.calibration import collect_images_from_dir, collect_images_from_split, sample_images

    if args.data_yaml:
        images = collect_images_from_split(args.data_yaml, split=args.cal_split)
    elif args.cal_images:
        images = collect_images_from_dir(args.cal_images)
    else:
        raise SystemExit("Provide either --cal-images or --data-yaml")

    if not images:
        raise SystemExit("No calibration images found")

    return sample_images(images, args.cal_sample_num, seed=args.cal_seed)


def to_container_path(path_value, workspace):
    relative = to_workspace_relative(path_value, workspace)
    return str(Path("/workspace") / relative)


def run_quantize(args):
    from tinyseg.calibration import write_calibration_tensors
    from tinyseg.rdk_x5_config import write_config

    workspace = Path(args.workspace).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mapper_script:
        print("--mapper-script is deprecated and ignored. TinySeg now generates config.yaml and calibration data internally.")

    onnx_host_path = Path(args.onnx).resolve()
    output_dir_host = output_dir.resolve()

    input_h, input_w = get_onnx_input_hw(onnx_host_path)
    model_stem = onnx_host_path.stem
    output_model_prefix = f"{model_stem}_bayese_{input_w}x{input_h}_nv12"

    selected_images = collect_calibration_images(args)
    workspace_dir = output_dir_host / ".quantize_workspace"
    cal_data_dir = workspace_dir / "calibration_data"
    bpu_output_dir = workspace_dir / "bpu_model_output"
    config_path = output_dir_host / "config.yaml"
    sources_path = output_dir_host / "calibration_sources.txt"
    mapper_log_path = output_dir_host / "hb_mapper_makertbin.log"

    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    bpu_output_dir.mkdir(parents=True, exist_ok=True)

    written_files = write_calibration_tensors(
        image_paths=selected_images,
        output_dir=cal_data_dir,
        target_hw=(input_h, input_w),
        preprocess=args.preprocess,
    )
    if not written_files:
        raise SystemExit("Calibration tensor generation produced no files")

    sources_path.write_text(
        "\n".join(str(path.resolve()) for path in selected_images) + "\n",
        encoding="utf-8",
    )

    write_config(
        config_path,
        onnx_model=to_container_path(onnx_host_path, workspace),
        cal_data_dir=to_container_path(cal_data_dir, workspace),
        working_dir=to_container_path(bpu_output_dir, workspace),
        output_model_file_prefix=output_model_prefix,
        jobs=args.jobs,
        optimize_level=args.optimize_level,
        quantized=args.quantized,
    )

    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{workspace}:/workspace",
        "-w",
        "/workspace",
        args.docker_image,
        "bash",
        "-lc",
        f"cd {to_container_path(workspace_dir, workspace)} && hb_mapper makertbin --config {to_container_path(config_path, workspace)} --model-type onnx",
    ]

    print("Running:")
    print(" ".join(command))
    subprocess.run(command, check=True)

    output_bin = bpu_output_dir / f"{output_model_prefix}.bin"
    if not output_bin.exists():
        raise SystemExit(f"Compiled model not found: {output_bin}")

    final_bin = output_dir_host / f"{output_model_prefix}.bin"
    if final_bin.exists():
        final_bin.unlink()
    shutil.move(str(output_bin), str(final_bin))

    generated_log = workspace_dir / "hb_mapper_makertbin.log"
    if generated_log.exists():
        if mapper_log_path.exists():
            mapper_log_path.unlink()
        shutil.move(str(generated_log), str(mapper_log_path))

    if not args.keep_workspace and workspace_dir.exists():
        shutil.rmtree(workspace_dir)

    print(f"Generated calibration tensors: {len(written_files)}")
    print(f"Saved config: {config_path}")
    print(f"Saved calibration source list: {sources_path}")
    print(f"Output bin: {final_bin}")
    if mapper_log_path.exists():
        print(f"Mapper log: {mapper_log_path}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_quantize(args)


if __name__ == "__main__":
    main()
