from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from tinyseg.data.labelme_drivable_stairs import CLASS_NAMES, run_conversion


def build_parser():
    parser = argparse.ArgumentParser(
        description="Build a reproducible YOLO-seg dataset from a YAML conversion recipe."
    )
    parser.add_argument("--config", required=True, help="Dataset conversion recipe YAML.")
    parser.add_argument(
        "--raw-root",
        default=None,
        help="Optional root prepended to relative input paths in the recipe.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output dataset root overriding the recipe output.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output dataset directory.",
    )
    return parser


def expand_value(value: Any) -> str:
    return os.path.expanduser(os.path.expandvars(str(value)))


def resolve_input_path(path_value: Any, raw_root: Path | None, recipe_dir: Path) -> Path:
    expanded = Path(expand_value(path_value))
    if expanded.is_absolute():
        return expanded.resolve()
    if raw_root is not None:
        return (raw_root / expanded).resolve()
    return (recipe_dir / expanded).resolve()


def find_repo_root(start_dir: Path) -> Path:
    for candidate in [start_dir, *start_dir.parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / "tinyseg").is_dir():
            return candidate
    return Path.cwd()


def resolve_output_path(path_value: Any, recipe_dir: Path) -> Path:
    expanded = Path(expand_value(path_value))
    if expanded.is_absolute():
        return expanded.resolve()
    repo_root = find_repo_root(recipe_dir)
    return (repo_root / expanded).resolve()


def list_from_config(value: Any, default: list[Any] | None = None) -> list[Any]:
    if value is None:
        return list(default or [])
    if isinstance(value, list):
        return value
    return [value]


def build_conversion_args(cli_args: argparse.Namespace):
    config_path = Path(cli_args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise SystemExit(f"recipe must be a YAML mapping: {config_path}")

    recipe_dir = config_path.parent
    raw_root_value = cli_args.raw_root or config.get("raw_root")
    raw_root = Path(expand_value(raw_root_value)).resolve() if raw_root_value else None

    input_values = list_from_config(config.get("inputs"))
    if not input_values:
        raise SystemExit("recipe must define at least one input path under `inputs`")
    input_paths = [resolve_input_path(path_value, raw_root, recipe_dir) for path_value in input_values]

    output_value = cli_args.output or config.get("output")
    if not output_value:
        raise SystemExit("recipe must define `output`, or pass --output")
    output_path = resolve_output_path(output_value, recipe_dir)

    classes = list_from_config(config.get("classes"), CLASS_NAMES)
    if classes != CLASS_NAMES:
        raise SystemExit(
            f"this converter currently expects classes {CLASS_NAMES}, got {classes}. "
            "Use the label alias map in tinyseg.data.labelme_drivable_stairs for new aliases."
        )

    split_config = config.get("split") or {}
    scan_config = config.get("scan") or {}
    options_config = config.get("options") or {}

    return SimpleNamespace(
        inputs=[str(path) for path in input_paths],
        output=str(output_path),
        val_ratio=float(split_config.get("val_ratio", config.get("val_ratio", 0.15))),
        split_mode=str(split_config.get("mode", config.get("split_mode", "temporal"))),
        seed=int(split_config.get("seed", config.get("seed", 42))),
        exclude_dir=list_from_config(scan_config.get("exclude_dir"), ["labelme", ".git", "__pycache__"]),
        include_path_regex=list_from_config(scan_config.get("include_path_regex")),
        exclude_path_regex=list_from_config(scan_config.get("exclude_path_regex")),
        skip_empty=bool(options_config.get("skip_empty", config.get("skip_empty", False))),
        overwrite=bool(cli_args.overwrite or options_config.get("overwrite", False)),
        yaml_path_mode=str(options_config.get("yaml_path_mode", "relative")),
        recipe_name=str(config.get("name", config_path.stem)),
        recipe_path=str(config_path),
    )


def main(argv=None):
    cli_args = build_parser().parse_args(argv)
    conversion_args = build_conversion_args(cli_args)
    run_conversion(conversion_args)


if __name__ == "__main__":
    main()
