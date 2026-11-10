import math
import os
import ssl

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import SYNTHETIC


def mixture_centers():
    angles = np.arange(8) * (2 * np.pi / 8)
    return 2 * np.column_stack((np.cos(angles), np.sin(angles)))


def sample_synthetic(name, n, rng):
    """Draw independently from the specified population, without a finite training split."""
    if name == "gaussian":
        points = rng.standard_normal((n, 2))
    elif name == "gaussian_mixture":
        points = mixture_centers()[rng.integers(0, 8, n)] + rng.normal(0, 0.1, (n, 2))
    elif name == "circle":
        theta = rng.uniform(0, 2 * math.pi, n)
        radius = 1 + rng.normal(0, 0.02, n)
        points = radius[:, None] * np.column_stack((np.cos(theta), np.sin(theta)))
    elif name == "sine":
        x = rng.uniform(-math.pi, math.pi, n)
        points = np.column_stack((x, np.sin(x) + rng.normal(0, 0.05, n)))
    else:
        raise ValueError(f"Unknown synthetic dataset {name!r}; expected {SYNTHETIC}")
    return np.asarray(points, dtype=np.float32)


def image_dataset(name, root, train=True, download=True):
    # Some python.org macOS installs omit a system CA file. Use a verified CA
    # bundle in that case, never disable TLS certificate verification.
    if ssl.get_default_verify_paths().cafile is None:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    from torchvision import datasets, transforms

    cls = {"mnist": datasets.MNIST, "fashion_mnist": datasets.FashionMNIST,
           "kmnist": datasets.KMNIST, "cifar10": datasets.CIFAR10}[name]
    transform = transforms.Compose([
        transforms.Resize((32, 32)), transforms.ToTensor(),
        transforms.Normalize((0.5,) * (3 if name == "cifar10" else 1),
                             (0.5,) * (3 if name == "cifar10" else 1)),
    ])
    try:
        return cls(root=root, train=train, download=download, transform=transform)
    except RuntimeError:
        if not download or name not in ("fashion_mnist", "kmnist"):
            raise
        # Official maintainers' HTTPS mirrors for legacy HTTP download failures.
        mirror = ("https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
                  if name == "fashion_mnist" else "https://codh.rois.ac.jp/kmnist/dataset/kmnist/")
        class HTTPSDataset(cls):
            mirrors = [mirror]
        # Torchvision derives its cache path from the class name.
        HTTPSDataset.__name__ = cls.__name__
        return HTTPSDataset(root=root, train=train, download=True, transform=transform)


def epoch_loader(dataset, config, seed, epoch):
    # Each epoch's permutation is independent of training RNG and can be rebuilt
    # exactly when resuming mid-epoch. Transforms intentionally have no randomness.
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                      drop_last=True, num_workers=config.num_workers,
                      generator=torch.Generator().manual_seed(seed + epoch),
                      pin_memory=False)
