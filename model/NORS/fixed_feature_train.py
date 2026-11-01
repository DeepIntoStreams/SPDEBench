"""Shared training loop for the fixed seven-channel Phi42 and KPZ features."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset


class CachedFeatureDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        self.features = features
        self.targets = targets
        self.indices = np.asarray(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = int(self.indices[index])
        features = torch.from_numpy(np.array(self.features[sample], copy=True))
        target = torch.from_numpy(np.array(self.targets[sample], copy=True))
        return features, target


def relative_l2_sum(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    batch = prediction.shape[0]
    error = torch.linalg.vector_norm(
        (prediction - target).reshape(batch, -1), dim=1
    )
    scale = torch.linalg.vector_norm(target.reshape(batch, -1), dim=1).clamp_min(
        1e-12
    )
    return (error / scale).sum()


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    accumulation_steps: int = 1,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    if training:
        optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
        for batch_index, (features, target) in enumerate(loader, start=1):
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            prediction = model(features)
            # t=0 is supplied through I_c[u0], so evaluate the learned dynamics only.
            loss = relative_l2_sum(prediction[:, 1:], target[:, 1:])
            if training:
                loss.backward()
                if batch_index % accumulation_steps == 0 or batch_index == len(loader):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach())
    return total / len(loader.dataset)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested) -> torch.device:
    if requested is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(requested))


def train_cached_features(
    cfg: DictConfig,
    *,
    features: np.ndarray,
    targets: np.ndarray,
    model: torch.nn.Module,
    feature_names: list[str],
    cache_path: Path,
    target_cache_path: Path | None,
    device: torch.device,
) -> None:
    """Split, train, and write metrics for one cached feature experiment."""
    if int(cfg.gradient_accumulation_steps) < 1:
        raise ValueError("gradient_accumulation_steps must be at least one.")
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total_requested = int(cfg.ntrain) + int(cfg.nval) + int(cfg.ntest)
    if total_requested > len(features):
        raise ValueError(
            f"Requested {total_requested} samples, but the cache has {len(features)}."
        )

    permutation = np.random.RandomState(int(cfg.seed)).permutation(len(features))
    train_indices = permutation[: int(cfg.ntrain)]
    val_indices = permutation[int(cfg.ntrain) : int(cfg.ntrain) + int(cfg.nval)]
    test_indices = permutation[
        int(cfg.ntrain) + int(cfg.nval) : int(cfg.ntrain) + int(cfg.nval) + int(cfg.ntest)
    ]
    split_path = output_dir / "split_indices.npz"
    np.savez(
        split_path,
        train=train_indices,
        validation=val_indices,
        test=test_indices,
    )
    loader_options = {
        "batch_size": int(cfg.batch_size),
        "num_workers": int(cfg.num_workers),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        CachedFeatureDataset(features, targets, train_indices),
        shuffle=True,
        **loader_options,
    )
    val_loader = DataLoader(
        CachedFeatureDataset(features, targets, val_indices),
        shuffle=False,
        **loader_options,
    )
    test_loader = DataLoader(
        CachedFeatureDataset(features, targets, test_indices),
        shuffle=False,
        **loader_options,
    )

    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(cfg.scheduler_factor),
        patience=int(cfg.scheduler_patience),
        threshold=float(cfg.scheduler_threshold),
        min_lr=1e-7,
    )
    checkpoint = output_dir / str(cfg.checkpoint_name)
    history: list[dict[str, float | int]] = []
    best_validation = float("inf")
    patience_reference = float("inf")
    stale_epochs = 0
    training_seconds = 0.0
    completed_epoch = 0
    stop_reason = "maximum_epochs"
    accumulation_steps = int(cfg.gradient_accumulation_steps)
    early_stopping = int(cfg.early_stopping)
    delta = float(cfg.delta)
    print_every = int(cfg.print_every)

    print(f"device={device}, parameters={count_parameters(model):,}")
    try:
        for epoch in range(1, int(cfg.epochs) + 1):
            start = time.perf_counter()
            train_loss = run_epoch(
                model,
                train_loader,
                device,
                optimizer,
                accumulation_steps=accumulation_steps,
            )
            training_seconds += time.perf_counter() - start
            validation_loss = run_epoch(model, val_loader, device, optimizer=None)
            scheduler.step(validation_loss)
            completed_epoch = epoch
            history.append(
                {"epoch": epoch, "train": train_loss, "validation": validation_loss}
            )

            if validation_loss < best_validation:
                best_validation = validation_loss
                torch.save(model.state_dict(), checkpoint)
            if validation_loss <= patience_reference - delta:
                patience_reference = validation_loss
                stale_epochs = 0
            else:
                stale_epochs += 1

            if epoch == 1 or epoch % print_every == 0:
                print(
                    f"epoch={epoch:04d} train={train_loss:.6f} "
                    f"val={validation_loss:.6f} "
                    f"lr={optimizer.param_groups[0]['lr']:.3e} "
                    f"early_stop={stale_epochs}/{early_stopping}"
                )
            if stale_epochs >= early_stopping:
                stop_reason = "early_stopping"
                print(
                    f"early stopping at epoch {epoch}; "
                    f"best validation={best_validation:.6f}"
                )
                break
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        print(
            f"\ninterrupted after epoch {completed_epoch}; "
            f"loading best checkpoint from {checkpoint}"
        )

    if not checkpoint.exists():
        raise RuntimeError("Training ended before a best checkpoint was saved.")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    train_loss = run_epoch(model, train_loader, device, optimizer=None)
    validation_loss = run_epoch(model, val_loader, device, optimizer=None)
    test_loss = run_epoch(model, test_loader, device, optimizer=None)
    metrics = {
        "train_relative_l2": train_loss,
        "validation_relative_l2": validation_loss,
        "test_relative_l2": test_loss,
        "best_validation_relative_l2": best_validation,
        "training_seconds": training_seconds,
        "completed_epoch": completed_epoch,
        "stop_reason": stop_reason,
        "parameters": count_parameters(model),
        "feature_names": list(feature_names),
        "feature_shape": list(features.shape),
        "target_shape": list(targets.shape),
        "data_path": str(Path(cfg.data_path).resolve()),
        "cache_path": str(Path(cache_path).resolve()),
        "split_path": str(split_path.resolve()),
        "arguments": OmegaConf.to_container(cfg, resolve=True),
        "history": history,
    }
    if target_cache_path is not None:
        metrics["target_cache_path"] = str(Path(target_cache_path).resolve())
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(
        json.dumps(
            {key: metrics[key] for key in metrics if "relative_l2" in key},
            indent=2,
        )
    )
    print(f"checkpoint={checkpoint}")
    print(f"metrics={metrics_path}")
