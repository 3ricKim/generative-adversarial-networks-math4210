import numpy as np

from .data import mixture_centers


def sliced_wasserstein(real, fake, projections=128, seed=2026):
    """Mean 1-Wasserstein distance across fixed unit projection directions."""
    real, fake = np.asarray(real), np.asarray(fake)
    if not isinstance(projections, int) or projections < 1:
        raise ValueError("projections must be a positive integer")
    if real.shape != fake.shape or real.ndim != 2 or len(real) < 2:
        raise ValueError("SWD requires equally sized (N, D) arrays, N >= 2")
    if not (np.isfinite(real).all() and np.isfinite(fake).all()):
        raise ValueError("Samples must be finite")
    directions = np.random.default_rng(seed).normal(size=(real.shape[1], projections))
    directions /= np.linalg.norm(directions, axis=0, keepdims=True)
    return float(np.abs(np.sort(real @ directions, axis=0) -
                        np.sort(fake @ directions, axis=0)).mean())


def mode_coverage(samples):
    samples = np.asarray(samples)
    if samples.ndim != 2 or samples.shape[1] != 2 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError("Mode coverage requires finite, nonempty (N, 2) samples")
    distances = np.linalg.norm(samples[:, None, :] - mixture_centers()[None, :, :], axis=2)
    nearest = distances.argmin(axis=1)
    valid = distances.min(axis=1) <= 0.3
    occupancy = np.bincount(nearest[valid], minlength=8) / len(samples)
    return {"mode_occupancy": occupancy.tolist(), "modes_covered": int((occupancy >= 0.01).sum()),
            "in_mode_fraction": float(valid.mean())}
