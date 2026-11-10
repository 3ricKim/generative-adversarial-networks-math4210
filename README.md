# GAN experiments for MATH 4210

A reproducible three-person project covering four simulated distributions and four image datasets. This repository includes PyTorch training/evaluation code, experiment configurations, a Colab launcher, and experimental results.

The local result set is complete: 16 baseline runs and the learning-rate follow-up, including 20 epochs on all four real datasets. Training used CPU for points and Apple MPS for images. The 19-test suite passed, including checkpoint recovery on CPU and MPS. See `results/comparison.md` and `results/analysis.md`. The Colab launcher is prepared for CUDA but was not executed in a signed-in cloud session here. The deliverables README lists the source and trained-model archive.

## Setup and smoke tests

Use Python 3.10+. First tested with Python 3.13, torch 2.9.1, torchvision 0.24.1, NumPy 2.2.6, Matplotlib 3.10.6, and PyYAML 6.0.3. Colab can retain its compatible preinstalled torch/torchvision pair. Every run records actual versions.

For the exact tested direct dependency versions, install `requirements-tested.txt` first, then install this project with `--no-deps`. Matching versions and seeds still do not guarantee identical results across different hardware.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/run_experiments.py --mode smoke
```

Smoke tests download the official datasets and run at most 20 steps each. They never count as completed baselines. Device `auto` chooses CUDA when available, MPS for images on Apple Silicon, and CPU for small synthetic models on this Mac.

## Full experiment matrix

| Configuration | Data | Budget | Seeds |
|---|---|---|---|
| gaussian | Standard bivariate Gaussian | 5,000 steps | 42, 43, 44 |
| gaussian_mixture | Eight Gaussians, radius 2, component SD 0.1 | 5,000 steps | 42, 43, 44 |
| circle | Unit circle, radial noise SD 0.02 | 5,000 steps | 42, 43, 44 |
| sine | x uniform on [−π, π], y = sin(x) + noise, SD 0.05 | 5,000 steps | 42, 43, 44 |
| mnist | Handwritten digits | 20 epochs | 42 |
| fashion_mnist | Clothing images | 20 epochs | 42 |
| kmnist | Kuzushiji characters | 20 epochs | 42 |
| cifar10 | Color object images | 20 epochs | 42 |

```bash
# All 16 baseline runs, followed by the lower-learning-rate mixture experiment:
python scripts/run_experiments.py --mode all

# Divide work by dataset:
python scripts/run_experiments.py --mode baseline --datasets gaussian gaussian_mixture circle sine
python scripts/run_experiments.py --mode baseline --datasets mnist fashion_mnist kmnist cifar10

# Requires completed, evaluated baseline matrix:
python scripts/run_experiments.py --mode followup
```

The runner skips completed training and resumes interrupted runs. Avoid concurrent jobs writing to the same run or downloading the same dataset. Points use 128/128-unit MLPs. Images use a 32×32 DCGAN with latent dimension 100 and hidden generator channels 256, 128, 64. Pixels map to [−1, 1]. Labels do not enter the GAN. Adam uses learning rate 0.0002 and betas (0.5, 0.999). Each step updates D and then G once using `BCEWithLogitsLoss` and a non-saturating G loss.

Official training splits use deterministic resizing, no augmentation, and shuffled batches. The last incomplete batch is dropped: 468 updates per epoch for the grayscale datasets, 390 for CIFAR-10. Official test images provide visual references only. Synthetic batches contain fresh population samples.

## Individual commands

```bash
python -m gan_project train --config configs/gaussian.yaml --seed 42 --output-dir runs/custom/gaussian42
python -m gan_project evaluate --run runs/custom/gaussian42
python -m gan_project summarize --runs runs --output-dir results

# Deliberate partial run, then continuation:
python -m gan_project train --config configs/gaussian.yaml --output-dir runs/recovery --max-steps 1000
python -m gan_project train --config configs/gaussian.yaml --output-dir runs/recovery --resume runs/recovery/checkpoints/latest.pt
```

`--max-steps` is an absolute step count. Keep configuration, seed, and smoke setting unchanged when resuming. Mid-epoch recovery recreates the batch permutation and skips processed batches. Exact continuation is tested on CPU. Different GPU/runtime versions can affect numerical reproducibility.

Each run stores configuration and environment metadata, source snapshots, per-step `losses.csv`, fixed-latent sample arrays, and `checkpoints/latest.pt`. Completed runs also have `final.pt`. Checkpoints include models, optimizers, progress, latent inputs, and Python/NumPy/torch RNG states. Load only trusted project checkpoint files.

Reevaluation creates comparison/progress/loss figures, synthetic distance plots, metrics, and sample arrays without retraining. It preserves a human `observations.md` worksheet. The summary includes failed, partial, smoke, and completed runs, checks for stale evaluations, and audits the 16 planned baseline runs. Shortened custom budgets do not satisfy that audit.

## Colab

Open `notebooks/colab_launcher.ipynb` in Colab and select a GPU runtime. Upload `deliverables/gan_project_source.zip` in its setup cell. This works before the code is pushed to GitHub. The notebook installs this package, mounts Drive through Colab's authorization UI, restores saved runs, and calls the same runner.

```bash
python scripts/run_experiments.py --mode all --device cuda --sync-dir /content/drive/MyDrive/math4210_gan/runs
```

Train under `/content`, copying artifacts to Drive at initialization, preview checkpoints, epoch boundaries, and completion. Restore durable runs before continuing in a fresh runtime. Work since the last copied checkpoint can be lost. Colab access and GPU allocation require the user's account/session.

## Evaluation and analysis

Synthetic SWD uses 2,048 samples, 128 fixed unit projections, and independent real reference draws. Mode coverage counts mixture components receiving at least 1% of all generated samples within radius 0.3. Real-real distances indicate finite-sample variation. Compare SWD within a dataset. Image conclusions are qualitative, and losses alone do not establish sample quality.

The three-person dataset requirement is four simulated and four real datasets. The architectures, metrics, seeds, and follow-up are project choices. See `docs/PROJECT_GUIDE.md` for the experiment workflow, `docs/THEORY.md` for the mathematical background, and `docs/SOURCES.md` for prior work and implementation attribution.

After all runs finish, `python scripts/package_results.py --include-checkpoints` creates a portable evidence ZIP with final trained models. Omitting that flag creates a smaller archive of figures, logs, source snapshots, and metrics. The notebook and code are packaged by `python scripts/package_source.py`.

Data, checkpoints, and private build files are ignored by Git. Preserve durable copies and per-run source archives.
