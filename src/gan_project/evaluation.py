import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid

from .config import Config
from .data import image_dataset, sample_synthetic
from .metrics import mode_coverage, sliced_wasserstein
from .models import build_models, noise
from .runtime import choose_device, load_checkpoint, write_json

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.titlesize": 13})


def save_figure(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def image_grid(samples, nrow=8, limit=64):
    tensor = torch.as_tensor(samples[:limit])
    return make_grid(tensor, nrow=nrow, padding=2, normalize=True, value_range=(-1, 1)).permute(1, 2, 0).numpy()


def evaluate(run_dir, device="cpu", checkpoint_path=None):
    run = Path(run_dir).resolve()
    metadata = json.loads((run / "metadata.json").read_text())
    path = Path(checkpoint_path) if checkpoint_path else run / "checkpoints" / (
        "final.pt" if metadata["status"] == "completed" else "latest.pt")
    state = load_checkpoint(path)
    config = Config(**state["config"]).validate()
    torch.set_num_threads(config.cpu_threads)
    selected = choose_device(device, config.synthetic)
    generator, _ = build_models(config)
    generator.to(selected).load_state_dict(state["generator"])
    generator.eval()
    fixed = state["fixed_z"].to(selected)
    with torch.inference_mode():
        fake = torch.cat([generator(z).cpu() for z in fixed.split(256)]).numpy()
    directory = run / "evaluation"
    directory.mkdir(exist_ok=True)
    result = {"dataset": config.dataset, "seed": state["seed"], "learning_rate": config.learning_rate,
              "status": metadata["status"], "smoke": state["smoke"], "step": state["step"],
              "epoch": state["epoch"], "planned_steps": metadata["planned_steps"],
              "elapsed_seconds": state["elapsed_seconds"], "checkpoint": str(path),
              "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "limitations": ("Three seeds assess variability, not a statistical guarantee." if config.synthetic else
                              "One training seed. Image quality and diversity require qualitative review.")}
    if config.synthetic:
        real = sample_synthetic(config.dataset, 2048, np.random.default_rng(2026))
        other = sample_synthetic(config.dataset, 2048, np.random.default_rng(2027))
        result.update(swd=sliced_wasserstein(real, fake), real_real_swd=sliced_wasserstein(real, other),
                      evaluation_samples=2048, projections=128)
        if config.dataset == "gaussian_mixture":
            result.update(mode_coverage(fake))
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
        for ax, samples, title in zip(axes, (real, fake), ("Reference population", f"Generated at step {state['step']}")):
            ax.scatter(samples[:, 0], samples[:, 1], s=4, alpha=0.45, color="#176B87")
            ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")
    else:
        dataset = image_dataset(config.dataset, config.data_root, train=False)
        indices = np.random.default_rng(2026).choice(len(dataset), size=64, replace=False)
        real = torch.stack([dataset[int(i)][0] for i in indices]).numpy()
        result["reference_split"] = "official test split (visual reference only)"
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, samples, title in zip(axes, (real, fake), ("Real test images", f"Generated after {state['epoch']} epochs / {state['step']} steps")):
            ax.imshow(image_grid(samples))
            ax.set_title(title)
            ax.axis("off")
    label = "SMOKE TEST" if state["smoke"] else ("COMPLETED RUN" if metadata["status"] == "completed" else "PARTIAL RUN")
    fig.suptitle(f"{config.dataset} · seed {state['seed']} · {label}")
    fig.tight_layout()
    save_figure(fig, directory / "comparison.png")
    np.savez_compressed(directory / "samples.npz", real=real, fake=fake)

    snapshots = [p for p in sorted((run / "samples").glob("step_*.npz"))
                 if int(p.stem.split("_")[1]) <= state["step"]]
    if snapshots:
        chosen = sorted(set(np.linspace(0, len(snapshots) - 1, min(4, len(snapshots))).astype(int)))
        fig, axes = plt.subplots(1, len(chosen), figsize=(4 * len(chosen), 4), squeeze=False)
        trajectory = []
        for sample_path in snapshots:
            with np.load(sample_path) as saved:
                if config.synthetic:
                    row = {"step": int(saved["step"]), "swd": sliced_wasserstein(real, saved["fake"])}
                    if config.dataset == "gaussian_mixture":
                        row.update(mode_coverage(saved["fake"]))
                    trajectory.append(row)
        for ax, index in zip(axes[0], chosen):
            with np.load(snapshots[index]) as saved:
                samples, sample_step, sample_epoch = saved["fake"], int(saved["step"]), int(saved["epoch"])
            if config.synthetic:
                ax.scatter(real[:, 0], real[:, 1], s=3, color="#AAB9C1", alpha=0.25)
                ax.scatter(samples[:, 0], samples[:, 1], s=3, color="#176B87", alpha=0.5)
                ax.set_aspect("equal", adjustable="datalim")
            else:
                ax.imshow(image_grid(samples))
                ax.axis("off")
            ax.set_title(f"Step {sample_step}" + (f" / epoch {sample_epoch}" if not config.synthetic else ""))
        fig.suptitle(f"{config.dataset} · same latent inputs · {label}")
        fig.tight_layout()
        save_figure(fig, directory / "progress.png")
        result["trajectory"] = trajectory
        if config.synthetic and trajectory:
            result["initial_swd"] = trajectory[0]["swd"]
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.plot([p["step"] for p in trajectory], [p["swd"] for p in trajectory], marker="o", label="Generated vs. reference")
            ax.axhline(result["real_real_swd"], color="#D47734", linestyle="--", label="Real vs. real")
            ax.set(xlabel="Training step", ylabel="Sliced Wasserstein distance", title=config.dataset)
            ax.legend()
            fig.tight_layout()
            save_figure(fig, directory / "distance.png")
    with (run / "losses.csv").open() as handle:
        losses = list(csv.DictReader(handle))
    # Ignore rows newer than the evaluated checkpoint.
    losses = [r for r in losses if int(r["step"]) <= state["step"]]
    if losses:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        steps = np.array([int(row["step"]) for row in losses])
        window = min(51, len(losses))
        for key, name, color in (("loss_g", "Generator", "#176B87"), ("loss_d", "Discriminator", "#D47734")):
            values = np.array([float(row[key]) for row in losses])
            ax.plot(steps, values, color=color, alpha=0.18, linewidth=0.5)
            ax.plot(steps[window-1:], np.convolve(values, np.ones(window)/window, mode="valid"), label=name, color=color)
        ax.set(xlabel="Training step", ylabel="Binary cross-entropy loss",
               title=f"{config.dataset} · {window}-step moving average · {label}")
        ax.legend()
        fig.tight_layout()
        save_figure(fig, directory / "losses.png")
    write_json(directory / "metrics.json", result)
    notes = run / "observations.md"
    if not notes.exists():
        notes.write_text(f"# {config.dataset}: experimental observations\n\n"
                         "Record only observations supported by this run's artifacts.\n\n"
                         "- Recognizable structure / distribution fit:\n- Diversity and missing modes:\n"
                         "- Training changes and their evidence:\n- Challenges actually encountered:\n"
                         "- Limitations:\n- Reviewer and review date:\n")
    return result
