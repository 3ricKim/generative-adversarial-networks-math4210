import hashlib
import json
import platform
import random
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import torch


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def choose_device(request="auto", synthetic=False):
    if request == "auto":
        # Tiny point models run faster on CPU than through the MPS dispatch path.
        request = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() and not synthetic else "cpu")
    if request == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable. Select a Colab GPU runtime.")
    if request == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but unavailable")
    return torch.device(request)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if state["mps"] is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


def environment(device):
    import torchvision
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return {"python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine(),
            "cuda": torch.version.cuda, "git_revision": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain")), "source_sha256": digest.hexdigest()}


def archive_source(output):
    root = Path(__file__).resolve().parents[2]
    # Preserve the original snapshot when resuming under a later source version.
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    target = output / ("source.zip" if not (output / "source.zip").exists() else
                       f"source_{digest.hexdigest()[:12]}.zip")
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        # Also works for a normal wheel install, where no src tree is present.
        for path in sorted(Path(__file__).parent.glob("*.py")):
            archive.write(path, f"src/gan_project/{path.name}")
        if (output / "config.json").exists():
            archive.write(output / "config.json", "run_config.json")
        for pattern in ("configs/*.yaml", "pyproject.toml", "docs/SOURCES.md"):
            for path in root.glob(pattern):
                archive.write(path, str(path.relative_to(root)))


def sync_run(output, destination):
    if not destination:
        return
    target = Path(destination).expanduser().resolve() / output.name
    if target == output or output in target.parents or target in output.parents:
        raise ValueError("Sync destination and local run directory must be separate")
    target.mkdir(parents=True, exist_ok=True)
    for source in output.rglob("*"):
        if source.is_file() and not source.name.endswith(".tmp"):
            dest = target / source.relative_to(output)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists() or source.stat().st_mtime_ns > dest.stat().st_mtime_ns:
                temp = dest.with_suffix(dest.suffix + ".tmp")
                shutil.copy2(source, temp)
                temp.replace(dest)


def load_checkpoint(path):
    # Checkpoints include Python/NumPy RNG states. Load ONLY trusted project files.
    return torch.load(path, map_location="cpu", weights_only=False)
