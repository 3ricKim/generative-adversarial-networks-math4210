import csv
import json
from pathlib import Path

import numpy as np

from .config import DATASETS, SYNTHETIC
from .runtime import write_json


def summarize(runs_dir, output_dir):
    root, output = Path(runs_dir).resolve(), Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, groups = [], {}
    for path in sorted(root.rglob("metadata.json")):
        metadata = json.loads(path.read_text())
        metrics_path = path.parent / "evaluation/metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        # Stale evaluations must not masquerade as final results after resuming.
        fresh = metrics.get("step") == metadata.get("step") and metrics.get("status") == metadata["status"]
        config = metadata["config"]
        baseline_budget = ((config["steps"] == 5000 and config["batch_size"] == 256)
                           if metadata["dataset"] in SYNTHETIC else
                           (config["epochs"] == 20 and config["batch_size"] == 128))
        notes = path.parent / "observations.md"
        review = notes.read_text() if notes.exists() else ""
        reviewed = bool(fresh and metrics.get("checkpoint_sha256") and metrics["checkpoint_sha256"] in review)
        paragraphs = [p.strip() for p in review.split("\n\n")
                      if p.strip() and not p.startswith(("#", "Review:", "Final checkpoint:", "Checkpoint", "-"))]
        observation = paragraphs[0] if reviewed and paragraphs else "Qualitative review pending"
        row = {"run": str(path.parent), "dataset": metadata["dataset"], "seed": metadata["seed"],
               "learning_rate": metadata["config"]["learning_rate"], "status": metadata["status"],
               "device": metadata["environment"]["device"],
               "smoke": metadata["smoke"], "step": metadata.get("step", 0),
               "planned_steps": metadata["planned_steps"], "epoch": metadata.get("epoch", 0),
               "training_budget": f"{metadata.get('step', 0)}/{metadata['planned_steps']} steps; {metadata.get('epoch', 0)} epochs",
               "elapsed_seconds": metadata.get("elapsed_seconds", 0), "evaluated": fresh,
               "baseline_budget": baseline_budget,
               "swd": metrics.get("swd", "") if fresh else "",
               "real_real_swd": metrics.get("real_real_swd", "") if fresh else "",
               "modes_covered": metrics.get("modes_covered", "") if fresh else "",
               "in_mode_fraction": metrics.get("in_mode_fraction", "") if fresh else "",
               "observations": observation, "observations_file": str(notes),
               "failure": metadata.get("error", "No execution error recorded; inspect observations for quality failures"),
               "limitations": metrics.get("limitations", "")}
        rows.append(row)
        if row["status"] == "completed" and not row["smoke"] and fresh:
            groups.setdefault((row["dataset"], row["learning_rate"]), []).append(row)
    columns = ["run", "dataset", "seed", "learning_rate", "status", "device", "smoke", "step", "planned_steps",
               "epoch", "training_budget", "elapsed_seconds", "evaluated", "baseline_budget", "swd", "real_real_swd", "modes_covered", "in_mode_fraction",
               "observations", "observations_file", "failure", "limitations"]
    with (output / "runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, columns)
        writer.writeheader()
        writer.writerows(rows)
    aggregates = []
    for (dataset, learning_rate), members in sorted(groups.items()):
        # Duplicated reruns of the same seed do not increase the replicate count.
        members = list({m["seed"]: m for m in members}.values())
        entry = {"dataset": dataset, "learning_rate": learning_rate,
                 "seeds": sorted(m["seed"] for m in members), "runs": len(members),
                 "devices": sorted({m["device"] for m in members}),
                 "runtime_seconds": sum(m["elapsed_seconds"] for m in members)}
        if dataset in SYNTHETIC:
            values = [m["swd"] for m in members]
            entry.update(swd_mean=float(np.mean(values)), swd_std=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                         swd_min=min(values), swd_max=max(values), real_real_swd=members[0]["real_real_swd"])
        if dataset == "gaussian_mixture":
            entry["modes_covered_by_seed"] = {m["seed"]: m["modes_covered"] for m in members}
            entry["in_mode_fraction_by_seed"] = {m["seed"]: m["in_mode_fraction"] for m in members}
        aggregates.append(entry)
    missing = []
    for dataset in DATASETS:
        seeds = {r["seed"] for r in groups.get((dataset, 0.0002), []) if r["baseline_budget"]}
        for seed in ((42, 43, 44) if dataset in SYNTHETIC else (42,)):
            if seed not in seeds:
                missing.append(f"{dataset}/seed_{seed}")
    report = {"baseline_complete": not missing, "missing_baselines": missing,
              "followup_complete": any(r["seed"] == 42 and r["baseline_budget"]
                                       for r in groups.get(("gaussian_mixture", 0.0001), [])),
              "aggregates": aggregates, "runs": rows}
    write_json(output / "summary.json", report)
    lines = ["# Experiment comparison", "", f"Baseline complete: **{not missing}**", "",
             "Smoke tests and partial or stale evaluations do not count toward completion.", "",
             "| Dataset | LR | Seeds | Device | SWD mean ± sample SD | Real-real SWD | Runtime (min) |",
             "|---|---:|---|---|---:|---:|---:|"]
    for row in aggregates:
        value = f"{row['swd_mean']:.4f}" if "swd_mean" in row else "Qualitative review"
        if row.get("swd_std") is not None:
            value += f" ± {row['swd_std']:.4f}"
        reference = f"{row['real_real_swd']:.4f}" if "real_real_swd" in row else "—"
        lines.append(f"| {row['dataset']} | {row['learning_rate']} | {row['seeds']} | {', '.join(row['devices'])} | {value} | {reference} | {row['runtime_seconds']/60:.1f} |")
    lines.extend(["", "## Remaining baselines", "", *[f"- {name}" for name in missing], "",
                  "## Interpretation", "", "Compare SWD within a dataset. Read observations.md alongside image grids. "
                  "Loss magnitude alone does not establish sample quality. Three synthetic seeds and one image seed limit generalization. "
                  "Runtime includes previews and checkpoint writes, excluding downloads and post-training evaluation. "
                  "Hardware and concurrent workloads differ, so runtimes do not isolate dataset complexity.", ""])
    (output / "comparison.md").write_text("\n".join(lines))
    return report
