import argparse
import json

from .config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reproducible MATH 4210 GAN experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train or resume a configured experiment")
    train.add_argument("--config", required=True)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    train.add_argument("--resume", help="Trusted project checkpoint")
    train.add_argument("--max-steps", type=int, help="Stop at this absolute step, marking the run partial")
    train.add_argument("--smoke", action="store_true", help="At most 20 steps, never a completed baseline")
    train.add_argument("--sync-dir", help="Parent folder for durable per-run checkpoint copies")
    evaluate = commands.add_parser("evaluate", help="Regenerate figures and metrics without retraining")
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    summarize = commands.add_parser("summarize", help="Compare runs and audit baseline completeness")
    summarize.add_argument("--runs", required=True)
    summarize.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            from .training import train
            result = train(load_config(args.config), args.seed, args.output_dir, args.device,
                           args.resume, args.max_steps, args.smoke, args.sync_dir)
        elif args.command == "evaluate":
            from .evaluation import evaluate
            result = evaluate(args.run, args.device)
        else:
            from .summary import summarize
            result = summarize(args.runs, args.output_dir)
        print(json.dumps({k: v for k, v in result.items() if k not in ("trajectory", "runs", "aggregates")}, indent=2))
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
