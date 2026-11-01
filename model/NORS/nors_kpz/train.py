"""Train derivative-feature NORS on a renormalized KPZ dataset."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data import build_feature_cache, feature_cache_path, load_kpz
from .features import FEATURE_NAMES
from .model import KPZNORS


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = (
    PROJECT_ROOT
    / "data_gen/results/KPZ+/KPZ+renosigma01_xi_lam005_trc128_1200.mat"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "model/result/NORS_KPZ"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--max-time", type=int, default=51)
    parser.add_argument("--sub-t", type=int, default=1)
    parser.add_argument("--sub-x", type=int, default=1)
    parser.add_argument("--spatial-points", type=int, default=128)
    parser.add_argument("--diffusion", type=float, default=1.0)
    parser.add_argument("--noise-scale", type=float, default=0.1)
    parser.add_argument("--renorm-constant", type=float, default=128.0)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--ntrain", type=int, default=840)
    parser.add_argument("--nval", type=int, default=180)
    parser.add_argument("--ntest", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--modes-x", type=int, default=16)
    parser.add_argument("--modes-t", type=int, default=16)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--time-padding", type=int, default=6)
    parser.add_argument("--scheduler-patience", type=int, default=50)
    parser.add_argument("--scheduler-factor", type=float, default=0.1)
    parser.add_argument("--scheduler-threshold", type=float, default=1e-6)
    parser.add_argument("--early-stopping", type=int, default=80)
    parser.add_argument("--delta", type=float, default=2e-4)
    parser.add_argument("--checkpoint-name", default="best_model.pth")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be at least one.")
    set_seed(args.seed)
    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_kpz(
        args.data_path,
        max_time=args.max_time,
        sub_t=args.sub_t,
        sub_x=args.sub_x,
        spatial_points=args.spatial_points,
    )
    total_requested = args.ntrain + args.nval + args.ntest
    if total_requested > len(data.noise):
        raise ValueError(
            f"Requested {total_requested} samples, but dataset has {len(data.noise)}."
        )

    cache_path = args.cache_path or feature_cache_path(
        args.output_dir / "features",
        args.data_path,
        sub_t=args.sub_t,
        sub_x=args.sub_x,
        spatial_points=args.spatial_points,
        noise_scale=args.noise_scale,
        renorm_constant=args.renorm_constant,
    )
    features = build_feature_cache(
        data,
        cache_path,
        device=device,
        batch_size=args.feature_batch_size,
        diffusion=args.diffusion,
        noise_scale=args.noise_scale,
        renorm_constant=args.renorm_constant,
        rebuild=args.rebuild_features,
    )
    print(f"features={features.shape}, channels={FEATURE_NAMES}")
    print(f"target={data.solution.shape}, cache={cache_path}")

    permutation = np.random.RandomState(args.seed).permutation(len(data.noise))
    train_indices = permutation[: args.ntrain]
    val_indices = permutation[args.ntrain : args.ntrain + args.nval]
    test_indices = permutation[
        args.ntrain + args.nval : args.ntrain + args.nval + args.ntest
    ]
    split_path = args.output_dir / "split_indices.npz"
    np.savez(
        split_path,
        train=train_indices,
        validation=val_indices,
        test=test_indices,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        CachedFeatureDataset(features, data.solution, train_indices),
        shuffle=True,
        **loader_options,
    )
    val_loader = DataLoader(
        CachedFeatureDataset(features, data.solution, val_indices),
        shuffle=False,
        **loader_options,
    )
    test_loader = DataLoader(
        CachedFeatureDataset(features, data.solution, test_indices),
        shuffle=False,
        **loader_options,
    )

    model = KPZNORS(
        num_features=len(FEATURE_NAMES),
        modes_x=args.modes_x,
        modes_t=args.modes_t,
        width=args.width,
        depth=args.depth,
        time_padding=args.time_padding,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        threshold=args.scheduler_threshold,
        min_lr=1e-7,
    )
    checkpoint = args.output_dir / args.checkpoint_name
    history: list[dict[str, float | int]] = []
    best_validation = float("inf")
    patience_reference = float("inf")
    stale_epochs = 0
    training_seconds = 0.0
    completed_epoch = 0
    stop_reason = "maximum_epochs"

    print(f"device={device}, parameters={count_parameters(model):,}")
    try:
        for epoch in range(1, args.epochs + 1):
            start = time.perf_counter()
            train_loss = run_epoch(
                model,
                train_loader,
                device,
                optimizer,
                accumulation_steps=args.gradient_accumulation_steps,
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
            if validation_loss <= patience_reference - args.delta:
                patience_reference = validation_loss
                stale_epochs = 0
            else:
                stale_epochs += 1

            if epoch == 1 or epoch % args.print_every == 0:
                print(
                    f"epoch={epoch:04d} train={train_loss:.6f} "
                    f"val={validation_loss:.6f} "
                    f"lr={optimizer.param_groups[0]['lr']:.3e} "
                    f"early_stop={stale_epochs}/{args.early_stopping}"
                )
            if stale_epochs >= args.early_stopping:
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
        "feature_names": list(FEATURE_NAMES),
        "feature_shape": list(features.shape),
        "target_shape": list(data.solution.shape),
        "data_path": str(args.data_path.resolve()),
        "cache_path": str(Path(cache_path).resolve()),
        "split_path": str(split_path.resolve()),
        "history": history,
    }
    metrics["arguments"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    metrics_path = args.output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(
        json.dumps(
            {key: value for key, value in metrics.items() if "relative_l2" in key},
            indent=2,
        )
    )
    print(f"checkpoint={checkpoint}")
    print(f"metrics={metrics_path}")


if __name__ == "__main__":
    main()
