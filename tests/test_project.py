import json

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from gan_project.config import Config, DATASETS, SYNTHETIC, load_config
from gan_project.data import mixture_centers, sample_synthetic
from gan_project.metrics import mode_coverage, sliced_wasserstein
from gan_project.models import build_models, noise
from gan_project.runtime import load_checkpoint
from gan_project.summary import summarize
from gan_project.training import train, train_step


@pytest.fixture(autouse=True)
def small_cpu():
    torch.set_num_threads(2)


@pytest.mark.parametrize("dataset", SYNTHETIC)
def test_sampling(dataset):
    samples = sample_synthetic(dataset, 20000, np.random.default_rng(5))
    assert samples.shape == (20000, 2) and samples.dtype == np.float32
    assert np.isfinite(samples).all()
    np.testing.assert_array_equal(samples, sample_synthetic(dataset, 20000, np.random.default_rng(5)))
    if dataset == "gaussian":
        np.testing.assert_allclose(samples.mean(axis=0), 0, atol=0.03)
        np.testing.assert_allclose(samples.std(axis=0), 1, atol=0.03)
    elif dataset == "circle":
        radius = np.linalg.norm(samples, axis=1)
        assert abs(radius.mean() - 1) < 0.002
        assert abs(radius.std() - 0.02) < 0.002
    elif dataset == "sine":
        assert np.max(np.abs(samples[:, 0])) <= np.pi
        assert abs((samples[:, 1] - np.sin(samples[:, 0])).std() - 0.05) < 0.002
    else:
        metrics = mode_coverage(samples)
        assert metrics["modes_covered"] == 8
        assert metrics["in_mode_fraction"] > 0.98


@pytest.mark.parametrize("dataset", ("gaussian", "mnist", "cifar10"))
def test_shapes_and_gradient_isolation(dataset):
    config = Config(dataset=dataset, batch_size=4)
    g, d = build_models(config)
    z = noise(config, 4, "cpu")
    fake = g(z)
    assert fake.shape == ((4, 2) if config.synthetic else (4, config.channels, 32, 32))
    assert d(fake).shape == (4,)
    if not config.synthetic:
        assert fake.min() >= -1 and fake.max() <= 1
    old_g = [p.detach().clone() for p in g.parameters()]
    old_d = [p.detach().clone() for p in d.parameters()]
    og = torch.optim.Adam(g.parameters(), lr=0.0002)
    od = torch.optim.Adam(d.parameters(), lr=0.0002)
    def before_d(*args):
        assert all(p.grad is None for p in g.parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in d.parameters())
    def before_g(*args):
        assert all(p.grad is None for p in d.parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in g.parameters())
    od.register_step_pre_hook(before_d)
    og.register_step_pre_hook(before_g)
    result = train_step(g, d, og, od, torch.randn_like(fake), z)
    assert all(np.isfinite(v) for v in result.values())
    assert any(not torch.equal(a, b) for a, b in zip(old_g, g.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(old_d, d.parameters()))
    assert all(p.grad is None for p in d.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in g.parameters())


def test_metrics():
    rng = np.random.default_rng(3)
    samples = rng.normal(size=(2048, 2))
    assert sliced_wasserstein(samples, samples) == 0
    independent = sliced_wasserstein(samples, rng.normal(size=(2048, 2)))
    assert 0 < independent < 0.15
    assert sliced_wasserstein(samples, samples + 4) > 2
    assert mode_coverage(np.repeat(mixture_centers(), 10, axis=0))["modes_covered"] == 8
    assert mode_coverage(np.repeat(mixture_centers()[:1], 80, axis=0))["modes_covered"] == 1
    assert mode_coverage(np.zeros((80, 2)))["modes_covered"] == 0
    with pytest.raises(ValueError):
        sliced_wasserstein(samples[:3], samples)


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_nested_equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("images", (False, True))
def test_exact_resume(tmp_path, monkeypatch, images):
    if images:
        dataset = TensorDataset(torch.linspace(-1, 1, 16 * 32 * 32).reshape(16, 1, 32, 32), torch.zeros(16))
        monkeypatch.setattr("gan_project.training.image_dataset", lambda *args: dataset)
    config = Config(dataset="mnist" if images else "circle", steps=8, epochs=2, batch_size=4, preview_every=2)
    full, split = tmp_path / "full", tmp_path / "split"
    train(config, 42, full, device="cpu")
    partial = train(config, 42, split, device="cpu", max_steps=3)
    assert partial["status"] == "paused" and not (split / "checkpoints/final.pt").exists()
    train(config, 42, split, device="cpu", resume=split / "checkpoints/latest.pt")
    a, b = [load_checkpoint(p / "checkpoints/final.pt") for p in (full, split)]
    for key in ("generator", "discriminator", "optimizer_g", "optimizer_d", "fixed_z", "step", "epoch", "batch_offset", "rng", "sampler_rng"):
        assert_nested_equal(a[key], b[key])
    with pytest.raises(ValueError, match="not empty"):
        train(config, 42, full)
    with pytest.raises(ValueError, match="must match"):
        train(config, 43, split, resume=split / "checkpoints/latest.pt")


def test_smoke_evaluate_summary(tmp_path):
    from gan_project.evaluation import evaluate
    run = tmp_path / "runs" / "smoke"
    config = Config(dataset="gaussian_mixture", steps=4, batch_size=8, preview_every=2)
    result = train(config, 42, run, device="cpu", smoke=True)
    assert result["status"] == "smoke_completed"
    a, b = evaluate(run), evaluate(run)
    assert a["swd"] == b["swd"]
    for name in ("comparison", "progress", "losses", "distance"):
        assert (run / f"evaluation/{name}.png").stat().st_size > 1000
    report = summarize(tmp_path / "runs", tmp_path / "summary")
    assert not report["baseline_complete"]
    assert len(report["missing_baselines"]) == 16
    assert report["aggregates"] == []


def test_config_validation(tmp_path):
    file = tmp_path / "bad.yaml"
    file.write_text("dataset: gaussian\nunknown: 1\n")
    with pytest.raises(ValueError, match="Unknown configuration"):
        load_config(file)
    for kwargs in ({"batch_size": 0}, {"learning_rate": -0.1}, {"beta1": 1}):
        with pytest.raises(ValueError):
            Config(dataset="gaussian", **kwargs).validate()


def test_shortened_and_stale_evaluations_do_not_complete_baseline(tmp_path):
    from gan_project.evaluation import evaluate
    run = tmp_path / "runs/short"
    train(Config(dataset="gaussian", steps=2, batch_size=8), 42, run, device="cpu")
    evaluate(run)
    report = summarize(tmp_path / "runs", tmp_path / "summary")
    assert not report["baseline_complete"]
    assert not report["runs"][0]["baseline_budget"]
    metadata_path = run / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["step"] += 1
    metadata_path.write_text(json.dumps(metadata))
    stale = summarize(tmp_path / "runs", tmp_path / "summary")
    assert not stale["runs"][0]["evaluated"]
    assert stale["aggregates"] == []


def test_durable_copy_recovery(tmp_path):
    import shutil
    config = Config(dataset="sine", steps=6, batch_size=8, preview_every=2)
    source, durable, recovered, full = [tmp_path / name for name in ("local", "drive", "recovered", "full")]
    train(config, 44, source, device="cpu", max_steps=3, sync_dir=durable)
    copied = durable / source.name
    assert load_checkpoint(copied / "checkpoints/latest.pt")["step"] == 3
    shutil.copytree(copied, recovered)
    train(config, 44, recovered, device="cpu", resume=recovered / "checkpoints/latest.pt")
    train(config, 44, full, device="cpu")
    a, b = [load_checkpoint(p / "checkpoints/final.pt") for p in (recovered, full)]
    assert_nested_equal(a["generator"], b["generator"])
    assert_nested_equal(a["optimizer_g"], b["optimizer_g"])
    with pytest.raises(ValueError, match="separate"):
        train(config, 44, tmp_path / "overlap", sync_dir=tmp_path)


@pytest.mark.parametrize("name", ("mnist", "fashion_mnist", "kmnist", "cifar10"))
def test_official_loader_preprocessing(monkeypatch, name):
    from PIL import Image
    from torchvision import datasets
    from gan_project.data import image_dataset
    class FixtureDataset:
        def __init__(self, root, train, download, transform):
            assert root == "fixture" and train is False and download is False
            self.transform = transform
        def __getitem__(self, index):
            array = np.full((28, 28, 3) if name == "cifar10" else (28, 28), 255, dtype=np.uint8)
            return self.transform(Image.fromarray(array)), 0
    for key in ("MNIST", "FashionMNIST", "KMNIST", "CIFAR10"):
        monkeypatch.setattr(datasets, key, FixtureDataset)
    sample, _ = image_dataset(name, "fixture", train=False, download=False)[0]
    assert sample.shape == (3 if name == "cifar10" else 1, 32, 32)
    torch.testing.assert_close(sample, torch.ones_like(sample))


@pytest.mark.skipif(not (torch.cuda.is_available() or torch.backends.mps.is_available()), reason="Accelerator unavailable")
def test_accelerator_resume(tmp_path, monkeypatch):
    dataset = TensorDataset(torch.linspace(-1, 1, 16 * 32 * 32).reshape(16, 1, 32, 32), torch.zeros(16))
    monkeypatch.setattr("gan_project.training.image_dataset", lambda *args: dataset)
    config = Config(dataset="mnist", epochs=2, batch_size=4)
    device = "cuda" if torch.cuda.is_available() else "mps"
    full, split = tmp_path / "full", tmp_path / "split"
    train(config, 42, full, device=device)
    train(config, 42, split, device=device, max_steps=3)
    train(config, 42, split, device=device, resume=split / "checkpoints/latest.pt")
    a, b = [load_checkpoint(p / "checkpoints/final.pt") for p in (full, split)]
    for model in ("generator", "discriminator"):
        for name in a[model]:
            torch.testing.assert_close(a[model][name], b[model][name], rtol=2e-6, atol=2e-6)
    for optimizer in ("optimizer_g", "optimizer_d"):
        for parameter in a[optimizer]["state"]:
            for key in a[optimizer]["state"][parameter]:
                torch.testing.assert_close(a[optimizer]["state"][parameter][key], b[optimizer]["state"][parameter][key], rtol=2e-6, atol=2e-6)
