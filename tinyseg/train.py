import argparse
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description="Train an Ultralytics YOLO segmentation model.")
    parser.add_argument("--data", required=True, help="Path to YOLO data.yaml.")
    parser.add_argument("--model", default="yolo11n-seg.pt", help="Base model or checkpoint path.")
    parser.add_argument("--epochs", type=int, default=120, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--device", default="0", help="Training device, for example 0, 0,1, or cpu.")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers.")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience.")
    parser.add_argument("--project", default="runs/seg", help="Ultralytics project directory.")
    parser.add_argument("--name", default="run", help="Ultralytics run name.")
    parser.add_argument("--cache", action="store_true", help="Enable dataset cache.")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted run.")
    return parser


def run_training(args):
    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        cache=args.cache,
        resume=args.resume,
        project=args.project,
        name=args.name,
    )

    save_dir = Path(results.save_dir).resolve()
    return {
        "run_dir": save_dir,
        "best_pt": save_dir / "weights" / "best.pt",
        "last_pt": save_dir / "weights" / "last.pt",
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    report = run_training(args)
    print("Training finished.")
    print(f"Run dir:  {report['run_dir']}")
    print(f"Best pt:  {report['best_pt']}")
    print(f"Last pt:  {report['last_pt']}")


if __name__ == "__main__":
    main()
