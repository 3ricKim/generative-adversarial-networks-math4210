"""Archive completed experimental evidence; optionally retain trained checkpoints."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from gan_project.summary import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--output", default="deliverables/experiment_evidence.zip")
    parser.add_argument("--include-checkpoints", action="store_true", help="Include final trained models for reuse")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    runs, output = Path(args.runs).resolve(), Path(args.output).resolve()
    report = summarize(runs, root / "results")
    if not report["baseline_complete"] or not report["followup_complete"]:
        raise SystemExit("Complete and evaluate the baseline matrix and follow-up before packaging final evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    files = []
    for row in report["runs"]:
        if row["status"] != "completed" or row["smoke"] or not row["evaluated"]:
            continue
        run = Path(row["run"])
        for file in sorted(run.rglob("*")):
            if not file.is_file() or file.name.endswith(".tmp"):
                continue
            if "checkpoints" in file.parts and (not args.include_checkpoints or file.name != "final.pt"):
                continue
            files.append((file, str(Path("runs") / file.relative_to(runs))))
    files.extend((file, str(Path("results") / file.name)) for file in (root / "results").iterdir() if file.is_file())
    manifest = {"includes_final_checkpoints": args.include_checkpoints,
                "baseline_runs": 16, "followup_runs": 1,
                "files": {name: hashlib.sha256(file.read_bytes()).hexdigest() for file, name in files}}
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for file, name in files:
            archive.write(file, name)
        archive.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(f"{output}: {len(files)} files, final checkpoints included={args.include_checkpoints}")


if __name__ == "__main__":
    main()
