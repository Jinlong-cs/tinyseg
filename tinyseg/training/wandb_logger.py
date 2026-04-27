from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tinyseg.data.yolo_dataset import collect_images_from_split


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_WANDB_KEY_FILE = ".wandb_api_key"
PALETTE = [
    (56, 114, 224),
    (46, 204, 113),
    (241, 196, 15),
    (231, 76, 60),
    (155, 89, 182),
    (26, 188, 156),
    (230, 126, 34),
    (52, 73, 94),
    (236, 240, 241),
]


def _normalize_tags(raw_tags: str | None) -> list[str] | None:
    if not raw_tags:
        return None
    tags = [tag.strip() for tag in raw_tags.split(",")]
    tags = [tag for tag in tags if tag]
    return tags or None


def _serialize(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    return str(value)


def _read_key_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    key = path.read_text(encoding="utf-8").strip()
    return key or None


def configure_wandb_credentials(api_key: str | None, key_file: str, workspace_root: Path) -> None:
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key.strip()
        return

    env_key = os.getenv("WANDB_API_KEY", "").strip()
    if env_key:
        return

    key_path = Path(key_file).expanduser()
    if not key_path.is_absolute():
        key_path = (workspace_root / key_path).resolve()
    file_key = _read_key_file(key_path)
    if file_key:
        os.environ["WANDB_API_KEY"] = file_key
        return

    raise SystemExit(
        "W&B is enabled but no API key was found. Set WANDB_API_KEY, pass --wandb-api-key, "
        f"or create a local {DEFAULT_WANDB_KEY_FILE} file."
    )


def _split_metrics(metrics: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    val_loss = {}
    val_metrics = {}
    for key, value in metrics.items():
        if key.startswith("val/"):
            val_loss[key] = value
        elif key == "fitness":
            val_metrics["val/fitness"] = value
        else:
            val_metrics[f"val/{key}"] = value
    return val_loss, val_metrics


def _prefix_metrics(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


def _is_rank0() -> bool:
    from ultralytics.utils import RANK

    return RANK in {-1, 0}


def _label_path_from_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def _resolve_names(names: Any) -> list[str]:
    if isinstance(names, dict):
        return [str(names[idx]) for idx in sorted(names)]
    if isinstance(names, (list, tuple)):
        return [str(item) for item in names]
    return []


def _iter_split_images(data_yaml: str, split: str) -> list[Path]:
    return [Path(path) for path in collect_images_from_split(data_yaml, split=split) if Path(path).suffix.lower() in IMAGE_EXTS]


def _put_title(image: np.ndarray, title: str) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 42), (0, 0, 0), thickness=-1)
    cv2.putText(image, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)


def _draw_label_overlay(image_path: Path, names: list[str], title: str) -> np.ndarray | None:
    label_path = _label_path_from_image(image_path)
    if not label_path.is_file() or label_path.stat().st_size == 0:
        return None

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return None

    height, width = image.shape[:2]
    overlay = image.copy()
    vis = image.copy()
    raw_lines = label_path.read_text(encoding="utf-8").splitlines()

    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line:
            continue
        try:
            values = [float(item) for item in line.split()]
        except ValueError:
            continue
        if len(values) < 5:
            continue

        class_id = int(values[0])
        color = PALETTE[class_id % len(PALETTE)]
        class_name = names[class_id] if 0 <= class_id < len(names) else str(class_id)
        coords = values[1:]

        if len(coords) >= 6 and len(coords) % 2 == 0:
            pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
            pts[:, 0] *= width
            pts[:, 1] *= height
            pts = np.round(pts).astype(np.int32)
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)
            anchor = pts[0]
            text_pos = (int(anchor[0]), max(60, int(anchor[1]) - 6))
        else:
            x_c, y_c, box_w, box_h = coords[:4]
            x1 = int(round((x_c - box_w / 2.0) * width))
            y1 = int(round((y_c - box_h / 2.0) * height))
            x2 = int(round((x_c + box_w / 2.0) * width))
            y2 = int(round((y_c + box_h / 2.0) * height))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness=2)
            text_pos = (x1, max(60, y1 - 6))

        cv2.putText(vis, class_name, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    vis = cv2.addWeighted(overlay, 0.30, vis, 0.70, 0.0)
    _put_title(vis, title)
    cv2.putText(vis, image_path.name, (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def _draw_prediction_overlay(image_path: Path, result: Any, names: list[str]) -> np.ndarray | None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return None

    overlay = image.copy()
    vis = image.copy()
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    class_ids = []
    confidences = []
    if boxes is not None and boxes.cls is not None:
        class_ids = [int(value) for value in boxes.cls.detach().cpu().numpy().tolist()]
    if boxes is not None and boxes.conf is not None:
        confidences = [float(value) for value in boxes.conf.detach().cpu().numpy().tolist()]

    polygons = []
    if masks is not None and masks.xy is not None:
        polygons = masks.xy

    if polygons:
        for index, polygon in enumerate(polygons):
            pts = np.asarray(polygon, dtype=np.int32)
            if len(pts) < 3:
                continue
            class_id = class_ids[index] if index < len(class_ids) else 0
            confidence = confidences[index] if index < len(confidences) else None
            color = PALETTE[class_id % len(PALETTE)]
            class_name = names[class_id] if 0 <= class_id < len(names) else str(class_id)
            label = f"{class_name} {confidence:.2f}" if confidence is not None else class_name
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)
            anchor = pts[0]
            text_pos = (int(anchor[0]), max(60, int(anchor[1]) - 6))
            cv2.putText(vis, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    elif boxes is not None and boxes.xyxy is not None:
        xyxy = boxes.xyxy.detach().cpu().numpy()
        for index, box in enumerate(xyxy):
            class_id = class_ids[index] if index < len(class_ids) else 0
            confidence = confidences[index] if index < len(confidences) else None
            color = PALETTE[class_id % len(PALETTE)]
            class_name = names[class_id] if 0 <= class_id < len(names) else str(class_id)
            label = f"{class_name} {confidence:.2f}" if confidence is not None else class_name
            x1, y1, x2, y2 = [int(round(value)) for value in box]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness=2)
            cv2.putText(vis, label, (x1, max(60, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    else:
        cv2.putText(vis, "no prediction", (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    vis = cv2.addWeighted(overlay, 0.30, vis, 0.70, 0.0)
    _put_title(vis, "Prediction")
    cv2.putText(vis, image_path.name, (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def _resize_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    if height == target_height:
        return image
    target_width = int(round(width * target_height / max(1, height)))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def _side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    target_height = min(left.shape[0], right.shape[0], 720)
    left = _resize_to_height(left, target_height)
    right = _resize_to_height(right, target_height)
    gap = np.full((target_height, 12, 3), 255, dtype=np.uint8)
    return np.concatenate([left, gap, right], axis=1)


def _draw_train_samples(data_yaml: str, names: list[str], sample_count: int) -> list[tuple[Path, np.ndarray]]:
    samples = []
    if sample_count <= 0:
        return samples

    for image_path in _iter_split_images(data_yaml, split="train"):
        sample = _draw_label_overlay(image_path, names, title="Train Ground Truth")
        if sample is None:
            continue
        samples.append((image_path, sample))
        if len(samples) >= sample_count:
            break

    return samples


def _normalize_imgsz(imgsz_values: Any) -> int | tuple[int, int]:
    if isinstance(imgsz_values, int):
        return imgsz_values
    values = list(imgsz_values)
    if len(values) == 1:
        return int(values[0])
    if len(values) == 2:
        return int(values[0]), int(values[1])
    raise ValueError("W&B prediction image size expects one value or two values: h w")


def _draw_prediction_samples(model_path: Path, data_yaml: str, names: list[str], sample_count: int, imgsz: int | tuple[int, int], device: str):
    if sample_count <= 0:
        return []

    image_paths = []
    gt_images = []
    for image_path in _iter_split_images(data_yaml, split="val"):
        ground_truth = _draw_label_overlay(image_path, names, title="Ground Truth")
        if ground_truth is None:
            continue
        image_paths.append(image_path)
        gt_images.append(ground_truth)
        if len(image_paths) >= sample_count:
            break

    if not image_paths:
        return []

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=imgsz,
        device=device,
        save=False,
        verbose=False,
    )

    samples = []
    for image_path, ground_truth, result in zip(image_paths, gt_images, results):
        prediction = _draw_prediction_overlay(image_path, result, names)
        if prediction is None:
            continue
        comparison = _side_by_side(ground_truth, prediction)
        samples.append((image_path, ground_truth, prediction, comparison))
    return samples


class TinySegWandbLogger:
    def __init__(self, args):
        self.project = args.wandb_project
        self.name = args.wandb_name or args.name
        self.tags = _normalize_tags(args.wandb_tags)
        self.resume = args.wandb_resume
        self.run_id = args.wandb_run_id
        self.sample_count = max(0, args.wandb_sample_count)
        pred_sample_count = args.wandb_pred_sample_count
        self.pred_sample_count = self.sample_count if pred_sample_count is None else max(0, pred_sample_count)
        self.pred_imgsz = _normalize_imgsz(args.wandb_pred_imgsz)
        self._sample_logged = False
        self._prediction_samples_logged = False
        self._wandb = None

    def _load_wandb(self):
        if self._wandb is None:
            try:
                import wandb
            except ImportError as exc:
                raise SystemExit("wandb is not installed. Run `uv sync` before using --wandb.") from exc
            self._wandb = wandb
        return self._wandb

    def _run(self):
        wb = self._load_wandb()
        return wb.run

    def _log(self, payload: dict[str, Any], step: int | None = None) -> None:
        if not payload or not _is_rank0():
            return
        run = self._run()
        if run is None:
            return
        self._load_wandb().log(payload, step=step)

    def on_pretrain_routine_end(self, trainer) -> None:
        if not _is_rank0():
            return

        wb = self._load_wandb()
        if wb.run is None:
            init_kwargs = {
                "project": self.project,
                "name": self.name,
                "config": _serialize(vars(trainer.args)),
                "dir": str(trainer.save_dir),
            }
            if self.tags:
                init_kwargs["tags"] = self.tags
            if self.run_id:
                init_kwargs["id"] = self.run_id
            if self.resume:
                init_kwargs["resume"] = self.resume
            wb.init(**init_kwargs)

        if self.sample_count > 0 and not self._sample_logged:
            samples = _draw_train_samples(
                trainer.args.data,
                _resolve_names(trainer.data.get("names", [])),
                self.sample_count,
            )
            if samples:
                table = wb.Table(columns=["image_path", "ground_truth"])
                for image_path, sample in samples:
                    table.add_data(str(image_path), wb.Image(sample))
                self._log({"train/ground_truth_samples": table}, step=0)
                self._sample_logged = True

    def on_train_epoch_end(self, trainer) -> None:
        train_loss = trainer.label_loss_items(trainer.tloss, prefix="train")
        train_metrics = _prefix_metrics(trainer.lr, "train")
        train_metrics["train/epoch"] = trainer.epoch + 1
        self._log({**train_loss, **train_metrics}, step=trainer.epoch + 1)

    def on_fit_epoch_end(self, trainer) -> None:
        val_loss, val_metrics = _split_metrics(trainer.metrics)
        self._log({**val_loss, **val_metrics}, step=trainer.epoch + 1)

    def _log_prediction_samples(self, trainer) -> None:
        if self.pred_sample_count <= 0 or self._prediction_samples_logged:
            return

        wb = self._load_wandb()
        model_path = trainer.best if trainer.best.exists() else getattr(trainer, "last", None)
        if model_path is None or not Path(model_path).exists():
            return

        try:
            samples = _draw_prediction_samples(
                model_path=Path(model_path),
                data_yaml=trainer.args.data,
                names=_resolve_names(trainer.data.get("names", [])),
                sample_count=self.pred_sample_count,
                imgsz=self.pred_imgsz,
                device=str(trainer.args.device),
            )
        except Exception as exc:
            print(f"W&B prediction sample logging failed: {exc}")
            return

        if not samples:
            return

        table = wb.Table(columns=["image_path", "ground_truth", "prediction", "comparison"])
        for image_path, ground_truth, prediction, comparison in samples:
            table.add_data(str(image_path), wb.Image(ground_truth), wb.Image(prediction), wb.Image(comparison))
        self._log({"val/prediction_samples": table}, step=trainer.epoch + 1)
        self._prediction_samples_logged = True

    def on_train_end(self, trainer) -> None:
        if not _is_rank0():
            return

        wb = self._load_wandb()
        if wb.run is None:
            return

        self._log_prediction_samples(trainer)

        if trainer.best.exists():
            artifact = wb.Artifact(name=f"{wb.run.id}-best", type="model")
            artifact.add_file(str(trainer.best), name="best.pt")
            wb.log_artifact(artifact, aliases=["best"])
        wb.finish()


def register_wandb_callbacks(model, args) -> None:
    logger = TinySegWandbLogger(args)
    model.add_callback("on_pretrain_routine_end", logger.on_pretrain_routine_end)
    model.add_callback("on_train_epoch_end", logger.on_train_epoch_end)
    model.add_callback("on_fit_epoch_end", logger.on_fit_epoch_end)
    model.add_callback("on_train_end", logger.on_train_end)
