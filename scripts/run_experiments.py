"""Run the experiment matrix sequentially. Completed runs are safe to revisit."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from gan_project.config import DATASETS, SYNTHETIC


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "baseline", "followup", "all"), default="smoke")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--sync-dir")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    root = Path(args.output_root).resolve()
    matrix = []
    if args.mode in ("smoke", "baseline", "all"):
        for dataset in args.datasets:
            for seed in ((42, 43, 44) if dataset in SYNTHETIC and args.mode != "smoke" else (42,)):
                matrix.append((dataset, seed, dataset, "smoke" if args.mode == "smoke" else "baseline"))
    if args.mode in ("followup", "all"):
        # Require a complete, evaluated baseline matrix before the follow-up.
        from gan_project.summary import summarize
        report = summarize(root, root / "summary")
        if args.mode == "followup" and not report["baseline_complete"]:
            parser.error("Finish and evaluate the baseline matrix before the follow-up")
        matrix.append(("gaussian_mixture", 42, "gaussian_mixture_lr0001", "followup"))
    for dataset, seed, config, kind in matrix:
        if kind == "followup":
            from gan_project.summary import summarize
            if not summarize(root, root / "summary")["baseline_complete"]:
                parser.error("Follow-up requires all 16 baseline runs to be complete and evaluated")
        run = root / kind / f"{config}_seed{seed}"
        metadata_path = run / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        finished = ((metadata.get("status") == "completed" and (run / "checkpoints/final.pt").exists()) or
                    (metadata.get("status") == "smoke_completed" and (run / "checkpoints/latest.pt").exists()))
        if not finished:
            command = [sys.executable, "-m", "gan_project", "train", "--config", str(project / "configs" / f"{config}.yaml"),
                       "--seed", str(seed), "--output-dir", str(run), "--device", args.device]
            if kind == "smoke":
                command += ["--smoke"]
            latest = run / "checkpoints/latest.pt"
            if latest.exists():
                command += ["--resume", str(latest)]
            if args.sync_dir:
                # Separate kind folders prevent smoke/baseline name collisions.
                command += ["--sync-dir", str(Path(args.sync_dir) / kind)]
            subprocess.run(command, cwd=project, check=True)
        subprocess.run([sys.executable, "-m", "gan_project", "evaluate", "--run", str(run)], cwd=project, check=True)
        if args.sync_dir:
            from gan_project.runtime import sync_run
            sync_run(run, str(Path(args.sync_dir) / kind))
    from gan_project.summary import summarize
    report = summarize(root, root / "summary")
    print(f"Baseline complete: {report['baseline_complete']}; missing: {report['missing_baselines']}")


if __name__ == "__main__":
    main()
