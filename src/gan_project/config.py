from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

SYNTHETIC = ("gaussian", "gaussian_mixture", "circle", "sine")
IMAGES = ("mnist", "fashion_mnist", "kmnist", "cifar10")
DATASETS = SYNTHETIC + IMAGES


@dataclass
class Config:
    dataset: str
    steps: int = 5000
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    preview_every: int = 500
    data_root: str = "data"
    num_workers: int = 0
    cpu_threads: int = 2

    @property
    def synthetic(self):
        return self.dataset in SYNTHETIC

    @property
    def latent_dim(self):
        return 2 if self.synthetic else 100

    @property
    def channels(self):
        return 3 if self.dataset == "cifar10" else 1

    def validate(self):
        if self.dataset not in DATASETS:
            raise ValueError(f"Unknown dataset {self.dataset!r}. Choose from {DATASETS}")
        for name in ("steps", "epochs", "batch_size", "preview_every", "cpu_threads"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.batch_size < 2:
            raise ValueError("batch_size must be at least 2 for image batch normalization")
        if not 0 < self.learning_rate < 1:
            raise ValueError("learning_rate must be between 0 and 1")
        if not (0 <= self.beta1 < 1 and 0 <= self.beta2 < 1):
            raise ValueError("Adam betas must be in [0, 1)")
        if self.num_workers < 0:
            raise ValueError("num_workers must be nonnegative")
        return self

    def to_dict(self):
        return asdict(self)


def load_config(path):
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping")
    unknown = raw.keys() - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    return Config(**raw).validate()
