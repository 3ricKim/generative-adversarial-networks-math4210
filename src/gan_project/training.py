import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import epoch_loader, image_dataset, sample_synthetic
from .models import build_models, noise
from .runtime import (archive_source, choose_device, environment, load_checkpoint,
                      restore_rng, rng_state, seed_everything, sync_run, write_json)


def train_step(generator, discriminator, optimizer_g, optimizer_d, real, z):
    """One D update and one non-saturating G update, with isolated gradients."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer_g.zero_grad(set_to_none=True)
    optimizer_d.zero_grad(set_to_none=True)
    fake = generator(z)
    real_logits, fake_logits = discriminator(real), discriminator(fake.detach())
    loss_d = criterion(real_logits, torch.ones_like(real_logits)) + criterion(
        fake_logits, torch.zeros_like(fake_logits))
    if not torch.isfinite(loss_d):
        raise FloatingPointError("Non-finite discriminator loss")
    loss_d.backward()
    optimizer_d.step()
    optimizer_d.zero_grad(set_to_none=True)
    discriminator.requires_grad_(False)
    try:
        output = discriminator(fake)
        loss_g = criterion(output, torch.ones_like(output))
        if not torch.isfinite(loss_g):
            raise FloatingPointError("Non-finite generator loss")
        loss_g.backward()
        optimizer_g.step()
    finally:
        discriminator.requires_grad_(True)
    # Transfer logged scalars together to avoid four GPU synchronization points.
    scalars = torch.stack((loss_g.detach(), loss_d.detach(), real_logits.detach().sigmoid().mean(),
                           fake_logits.detach().sigmoid().mean())).cpu().tolist()
    return dict(zip(("loss_g", "loss_d", "d_real", "d_fake"), scalars))


def train(config, seed, output_dir, device="auto", resume=None, max_steps=None,
          smoke=False, sync_dir=None):
    config.validate()
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    output = Path(output_dir).expanduser().resolve()
    if sync_dir:
        target = Path(sync_dir).expanduser().resolve() / output.name
        if target == output or output in target.parents or target in output.parents:
            raise ValueError("Sync destination must be separate from the local run directory")
    checkpoint = load_checkpoint(resume) if resume else None
    if checkpoint:
        saved = checkpoint["config"].copy()
        current = config.to_dict()
        for key in ("data_root", "num_workers", "cpu_threads"):
            saved.pop(key)
            current.pop(key)
        if saved != current or checkpoint["seed"] != seed or checkpoint["smoke"] != smoke:
            raise ValueError("Resume configuration, seed, and smoke flag must match the checkpoint")
    elif output.exists() and any(p.is_file() for p in output.rglob("*")):
        raise ValueError(f"Run directory is not empty: {output}. Choose another directory or --resume.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "samples").mkdir(exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    torch.set_num_threads(config.cpu_threads)
    selected = choose_device(device, config.synthetic)
    seed_everything(seed)
    generator, discriminator = [m.to(selected) for m in build_models(config)]
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=config.learning_rate,
                                   betas=(config.beta1, config.beta2))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config.learning_rate,
                                   betas=(config.beta1, config.beta2))
    sampler = np.random.default_rng(seed)
    fixed_z = noise(config, 2048 if config.synthetic else 64, selected,
                    torch.Generator().manual_seed(seed + 10000))
    step, epoch, batch_offset, previous_seconds = 0, 0, 0, 0.0
    if checkpoint:
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        step, epoch, batch_offset = checkpoint["step"], checkpoint["epoch"], checkpoint["batch_offset"]
        previous_seconds = checkpoint["elapsed_seconds"]
        sampler.bit_generator.state = checkpoint["sampler_rng"]
        fixed_z = checkpoint["fixed_z"].to(selected)
    dataset = None if config.synthetic else image_dataset(config.dataset, config.data_root)
    if dataset is not None and len(dataset) < config.batch_size:
        raise ValueError("Training split must contain at least one full batch")
    planned_steps = config.steps if config.synthetic else (len(dataset) // config.batch_size) * config.epochs
    limit = min(planned_steps, max_steps) if max_steps else planned_steps
    if smoke:
        limit = min(limit, 20)
    metadata = {"dataset": config.dataset, "seed": seed, "config": config.to_dict(),
                "environment": environment(selected), "smoke": smoke,
                "planned_steps": planned_steps, "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "training_size": len(dataset) if dataset is not None else "population sampling",
                "resumed_from": str(Path(resume).resolve()) if resume else None}
    if checkpoint:
        metadata["original_environment"] = checkpoint["metadata"]["environment"]
    write_json(output / "config.json", config.to_dict())
    archive_source(output)
    write_json(output / "metadata.json", metadata)
    if checkpoint:
        restore_rng(checkpoint["rng"])
    started = time.monotonic()

    def elapsed():
        return previous_seconds + time.monotonic() - started

    def preview():
        was_training = generator.training
        generator.eval()
        with torch.inference_mode():
            # Chunking keeps evaluation memory predictable on smaller GPUs.
            samples = torch.cat([generator(z).cpu() for z in fixed_z.split(256)])
        generator.train(was_training)
        np.savez_compressed(output / "samples" / f"step_{step:07d}.npz",
                            fake=samples.numpy(), step=step, epoch=epoch)

    def save():
        metadata.update(step=step, epoch=epoch, batch_offset=batch_offset, elapsed_seconds=elapsed())
        state = {"format_version": 1, "config": config.to_dict(), "seed": seed, "smoke": smoke,
                 "generator": generator.state_dict(), "discriminator": discriminator.state_dict(),
                 "optimizer_g": optimizer_g.state_dict(), "optimizer_d": optimizer_d.state_dict(),
                 "step": step, "epoch": epoch, "batch_offset": batch_offset,
                 "elapsed_seconds": elapsed(), "rng": rng_state(),
                 "sampler_rng": sampler.bit_generator.state, "fixed_z": fixed_z.cpu(),
                 "metadata": metadata}
        path = output / "checkpoints" / "latest.pt"
        temporary = path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(path)
        write_json(output / "metadata.json", metadata)
        sync_run(output, sync_dir)

    # Reconcile a CSV copied after its checkpoint (e.g. interruption during sync).
    log_path = output / "losses.csv"
    columns = ["step", "epoch", "loss_g", "loss_d", "d_real", "d_fake", "elapsed_seconds"]
    existing = []
    if checkpoint and log_path.exists():
        with log_path.open() as handle:
            existing = [r for r in csv.DictReader(handle) if int(r["step"]) <= step]
    with log_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(existing)
        handle.flush()
        try:
            if step == 0:
                preview()
                save()
            while step < limit:
                loader = None if config.synthetic else epoch_loader(dataset, config, seed, epoch)
                batches = range(limit - step) if config.synthetic else enumerate(loader)
                for batch in batches:
                    if config.synthetic:
                        real = torch.from_numpy(sample_synthetic(config.dataset, config.batch_size, sampler)).to(selected)
                    else:
                        index, (real, _) = batch
                        if index < batch_offset:
                            continue
                        real = real.to(selected)
                    values = train_step(generator, discriminator, optimizer_g, optimizer_d,
                                        real, noise(config, len(real), selected))
                    step += 1
                    if not config.synthetic:
                        batch_offset = index + 1
                        if batch_offset == len(loader):
                            epoch += 1
                            batch_offset = 0
                    writer.writerow({"step": step, "epoch": epoch, **values, "elapsed_seconds": elapsed()})
                    boundary = (step % config.preview_every == 0 if config.synthetic else batch_offset == 0)
                    if boundary or step >= limit:
                        preview()
                        handle.flush()
                        save()
                        print(f"{config.dataset} seed={seed} step={step}/{planned_steps} epoch={epoch} "
                              f"G={values['loss_g']:.3f} D={values['loss_d']:.3f} {elapsed():.1f}s", flush=True)
                    if step >= limit:
                        break
            metadata["status"] = "smoke_completed" if smoke else ("completed" if step == planned_steps else "paused")
            # latest.pt remains the resumable canonical checkpoint. final.pt is a
            # distinct completed artifact, never created for partial/smoke runs.
            save()
            if metadata["status"] == "completed":
                import shutil
                shutil.copy2(output / "checkpoints/latest.pt", output / "checkpoints/final.pt")
                sync_run(output, sync_dir)
        except BaseException as error:
            metadata.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                            error=f"{type(error).__name__}: {error}", step=step, elapsed_seconds=elapsed())
            write_json(output / "metadata.json", metadata)
            sync_run(output, sync_dir)
            raise
    return metadata
