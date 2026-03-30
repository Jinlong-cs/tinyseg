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
    parser.add_argument("--plots", action="store_true", help="Save Ultralytics training plots.")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted run.")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging.")
    parser.add_argument("--wandb-project", default="tinyseg", help="W&B project name.")
    parser.add_argument("--wandb-name", default=None, help="Optional W&B run name. Defaults to --name.")
    parser.add_argument("--wandb-tags", default=None, help="Comma-separated W&B tags.")
    parser.add_argument("--wandb-resume", default=None, help="Optional W&B resume mode.")
    parser.add_argument("--wandb-run-id", default=None, help="Optional fixed W&B run id.")
    parser.add_argument(
        "--wandb-api-key",
        default=None,
        help="Optional W&B API key. Prefer using WANDB_API_KEY or a local .wandb_api_key file instead.",
    )
    parser.add_argument(
        "--wandb-key-file",
        default=".wandb_api_key",
        help="Local gitignored file that stores the W&B API key.",
    )
    return parser


def run_training(args):
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS
    repo_root = Path(__file__).resolve().parents[1]

    original_wandb_setting = SETTINGS.get("wandb", True)
    SETTINGS["wandb"] = False

    if args.wandb:
        from tinyseg.wandb_logger import configure_wandb_credentials, register_wandb_callbacks

        configure_wandb_credentials(
            api_key=args.wandb_api_key,
            key_file=args.wandb_key_file,
            workspace_root=repo_root,
        )

    model = YOLO(args.model)
    if args.wandb:
        register_wandb_callbacks(model, args)

    try:
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            patience=args.patience,
            cache=args.cache,
            plots=args.plots,
            resume=args.resume,
            project=args.project,
            name=args.name,
        )
    finally:
        SETTINGS["wandb"] = original_wandb_setting

    save_dir = Path(results.save_dir).resolve()
    return {
        "run_dir": save_dir,
        "best_pt": save_dir / "weights" / "best.pt",
        "last_pt": save_dir / "weights" / "last.pt",
        "wandb_enabled": args.wandb,
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
