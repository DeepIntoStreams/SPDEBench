"""Evaluate a Phi42 NORS checkpoint on independent renormalized MAT datasets."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from timeit import default_timer

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from evaluations.metrics import (
    CrossCorrelationMetric,
    HsLossMetric,
    LpLossMetric,
    SigW1Metric,
)

from .data import Phi42Data, build_feature_cache, feature_cache_path, load_phi42
from .features import FEATURE_NAMES
from .model import Phi42NORS
from .train import count_parameters, relative_l2_sum


SUMMARY_KEYS = [
    "loss_test",
    "test_Rel_L2",
    "test_W1_2",
    "test_SigW1",
    "test_AutoCorr",
    "test_CrossCorr",
    "inference_avg_sample_time",
    "inference_throughput",
    "feature_avg_sample_time",
    "end_to_end_avg_sample_time",
]


class EvaluationDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = features
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.from_numpy(np.array(self.features[index], copy=True))
        target = torch.from_numpy(np.array(self.targets[index], copy=True))
        return features, target


def take_last(data: Phi42Data, size: int) -> Phi42Data:
    if size > len(data.noise):
        raise ValueError(f"Requested {size} samples, but the dataset contains {len(data.noise)}.")
    start = len(data.noise) - size
    return Phi42Data(
        noise=data.noise[start:],
        solution=data.solution[start:],
        time=data.time,
        x=data.x,
        y=data.y,
        source_path=data.source_path,
        solution_key=data.solution_key,
    )


def make_loader(data_path: str | Path, args, device: torch.device):
    data = take_last(
        load_phi42(
            data_path,
            max_time=args.T,
            sub_t=args.sub_t,
            sub_x=args.sub_x,
        ),
        args.eval_size,
    )
    cache_path = feature_cache_path(
        args.feature_cache_dir,
        data_path,
        sub_t=args.sub_t,
        sub_x=args.sub_x,
        noise_scale=args.noise_scale,
    )
    cache_existed = cache_path.exists() and cache_path.with_suffix(".json").exists()
    start = default_timer()
    features = build_feature_cache(
        data,
        cache_path,
        device=device,
        batch_size=args.feature_batch_size,
        diffusion=args.diffusion,
        noise_scale=args.noise_scale,
        rebuild=args.rebuild_features,
    )
    feature_seconds = default_timer() - start
    loader = DataLoader(
        EvaluationDataset(features, data.solution),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    return data, loader, cache_path, feature_seconds, cache_existed and not args.rebuild_features


def collect_predictions(model, loader, device):
    model.eval()
    reals, predictions = [], []
    with torch.no_grad():
        for features, target in loader:
            prediction = model(features.to(device))
            # Unified evaluation convention: [B,X,Y,T].
            predictions.append(prediction.permute(0, 2, 3, 1).detach().cpu())
            reals.append(target.permute(0, 2, 3, 1).detach().cpu())
    return torch.cat(reals, dim=0), torch.cat(predictions, dim=0)


def evaluate_relative_l2(model, loader, device) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for features, target in loader:
            features = features.to(device)
            target = target.to(device)
            prediction = model(features)
            total += float(relative_l2_sum(prediction[:, 1:], target[:, 1:]))
    return total / len(loader.dataset)


def to_path_tensor(value):
    if value.ndim == 5 and value.shape[-1] == 1:
        value = value[..., 0]
    if value.ndim != 4:
        raise ValueError(f"Expected [B,X,Y,T], got {tuple(value.shape)}")
    batch, x, y, time = value.shape
    return value.permute(0, 3, 1, 2).reshape(batch, time, x * y)


def stable_autocorr(value, max_lag=64, eps=1e-8):
    max_lag = min(max_lag, value.shape[1])
    value = value - value.mean((0, 1), keepdim=True)
    variance = torch.var(value, unbiased=False, dim=(0, 1))
    values = []
    for lag in range(max_lag):
        product = value[:, lag:] * value[:, :-lag] if lag > 0 else value.square()
        correlation = torch.mean(product, dim=(0, 1)) / (variance + eps)
        values.append(torch.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0))
    return torch.stack(values)


def stable_autocorr_metric(real_path, prediction_path, max_lag=64):
    real = stable_autocorr(real_path, max_lag=max_lag)
    prediction = stable_autocorr(prediction_path, max_lag=max_lag)
    return (prediction - real.to(prediction.device)).square().mean().sqrt()


def block_crosscorr(real_path, prediction_path, block_size=128, lags=64):
    feature_count = real_path.shape[-1]
    lags = min(lags, real_path.shape[1])
    weighted_total = 0.0
    total_pairs = 0
    for start in range(0, feature_count, block_size):
        end = min(start + block_size, feature_count)
        block_features = end - start
        if block_features <= 0:
            continue
        value = CrossCorrelationMetric(lags=lags).measure(
            real_path[..., start:end],
            prediction_path[..., start:end],
        )
        value = torch.nan_to_num(torch.as_tensor(value), nan=0.0, posinf=0.0, neginf=0.0)
        pair_count = block_features * (block_features + 1) // 2
        weighted_total += float(value) * pair_count
        total_pairs += pair_count
    if total_pairs == 0:
        raise ValueError("No features available for block CrossCorr.")
    return torch.tensor(weighted_total / total_pairs)


def evaluate_metrics(model, loader, device, prefix, crosscorr_block_size=128, max_lag=64):
    real, prediction = collect_predictions(model, loader, device)
    real_path = to_path_tensor(real)
    prediction_path = to_path_tensor(prediction)
    real_signature = real_path.permute(0, 2, 1)
    prediction_signature = prediction_path.permute(0, 2, 1)
    metric_specs = [
        ("Rel_L2", lambda: LpLossMetric(mode="rel").measure(real, prediction)),
        ("W1_2", lambda: HsLossMetric(k=1).measure(real, prediction)),
        ("SigW1", lambda: SigW1Metric().measure(real_signature, prediction_signature)),
        (
            "AutoCorr",
            lambda: stable_autocorr_metric(real_path, prediction_path, max_lag=max_lag),
        ),
        (
            "CrossCorr",
            lambda: block_crosscorr(
                real_path,
                prediction_path,
                block_size=crosscorr_block_size,
                lags=max_lag,
            ),
        ),
    ]
    print(f"{prefix} metrics:", flush=True)
    results = {}
    for name, compute_metric in metric_specs:
        try:
            print(f"computing {prefix}_{name}...", flush=True)
            value = compute_metric()
            results[name] = float(value)
            print(f"{prefix}_{name}: {results[name]:.6f}", flush=True)
        except Exception as error:
            results[name] = None
            print(f"{prefix}_{name}: failed ({error})", flush=True)
    return results, real, prediction


def measure_inference_time(model, loader, device, warmup_batches=1):
    was_training = model.training
    model.eval()
    total_time = 0.0
    total_batches = 0
    total_samples = 0
    with torch.no_grad():
        for batch_index, (features, _) in enumerate(loader):
            features = features.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = default_timer()
            _ = model(features)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = default_timer() - start
            if batch_index >= warmup_batches:
                total_time += elapsed
                total_batches += 1
                total_samples += features.shape[0]
    if was_training:
        model.train()
    return {
        "total_time": total_time,
        "num_batches": total_batches,
        "num_samples": total_samples,
        "avg_batch_time": total_time / total_batches if total_batches else float("nan"),
        "avg_sample_time": total_time / total_samples if total_samples else float("nan"),
        "throughput": total_samples / total_time if total_time > 0 else float("nan"),
    }


def build_model(args, device):
    model = Phi42NORS(
        num_features=len(FEATURE_NAMES),
        modes_x=args.modes1,
        modes_y=args.modes2,
        modes_t=args.modes3,
        width=args.width,
        depth=args.layers,
        time_padding=args.time_padding,
    ).to(device)
    print(f"The model has {count_parameters(model)} parameters")
    return model


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = (
        checkpoint.get("model_state_dict", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    model.load_state_dict(state_dict)
    print(f"loaded checkpoint: {checkpoint_path}")
    if isinstance(checkpoint, dict) and "epoch" in checkpoint and "best_val_loss" in checkpoint:
        print(
            f"checkpoint epoch: {checkpoint['epoch']} | "
            f"best val loss: {checkpoint['best_val_loss']:.6f}"
        )
    return checkpoint


def evaluate_one(args, model, data_path, index, device):
    data, loader, cache_path, feature_seconds, cache_hit = make_loader(data_path, args, device)
    print("\n" + "=" * 80)
    print(f"Evaluation dataset {index}: {data_path}")
    print(f"W shape: {tuple(data.noise.shape)}")
    print(f"sol shape: {tuple(data.solution.shape)}")
    print(f"features: {cache_path} ({'reused' if cache_hit else 'built'})")
    print(f"Evaluating samples: {len(loader.dataset)}")

    loss_test = evaluate_relative_l2(model, loader, device)
    inference = measure_inference_time(
        model,
        loader,
        device,
        warmup_batches=args.warmup_batches,
    )
    metrics, real, prediction = evaluate_metrics(
        model,
        loader,
        device,
        prefix=f"test_dataset_{index}",
        crosscorr_block_size=args.crosscorr_block_size,
        max_lag=args.max_lag,
    )
    prediction_path = None
    if not args.skip_save_predictions:
        prediction_path = args.output_dir / f"predictions_dataset_{index}.pt"
        torch.save(
            {
                "u_real": real,
                "u_pred": prediction,
                "data_path": str(data_path),
            },
            prediction_path,
        )

    feature_avg = feature_seconds / len(loader.dataset)
    inference_avg = inference.get("avg_sample_time")
    result = {
        "dataset": str(data_path),
        "num_samples": len(loader.dataset),
        "feature_cache": str(cache_path),
        "feature_cache_hit": cache_hit,
        "feature_cache_deleted": False,
        "prediction_file": str(prediction_path) if prediction_path is not None else None,
        "loss_test": loss_test,
        "test_Rel_L2": metrics.get("Rel_L2"),
        "test_W1_2": metrics.get("W1_2"),
        "test_SigW1": metrics.get("SigW1"),
        "test_AutoCorr": metrics.get("AutoCorr"),
        "test_CrossCorr": metrics.get("CrossCorr"),
        "inference_avg_sample_time": inference_avg,
        "inference_throughput": inference.get("throughput"),
        "inference_avg_batch_time": inference.get("avg_batch_time"),
        "inference_total_time": inference.get("total_time"),
        "feature_total_time": feature_seconds,
        "feature_avg_sample_time": feature_avg,
        "end_to_end_avg_sample_time": feature_avg + inference_avg,
    }
    print("dataset result:")
    for key in SUMMARY_KEYS:
        value = result.get(key)
        if value is not None:
            print(f"{key}: {value:.6f}")
    if args.delete_feature_cache_after_eval:
        del loader
        gc.collect()
        cache_path.unlink(missing_ok=True)
        cache_path.with_suffix(".json").unlink(missing_ok=True)
        result["feature_cache_deleted"] = True
        print(f"deleted feature cache: {cache_path}")
    return result


def summarize(results):
    summary = {}
    print("\n" + "=" * 80)
    print("Independent test dataset summary (mean +/- std):")
    for key in SUMMARY_KEYS:
        values = np.asarray(
            [row[key] for row in results if row.get(key) is not None],
            dtype=np.float64,
        )
        if values.size == 0:
            continue
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if values.size > 1 else 0.0
        summary[key] = {"mean": mean, "std": std}
        print(f"{key}: {mean:.6f} +/- {std:.6f}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate NORS Phi42 checkpoint on independent MAT test datasets"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-data-paths", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path)
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--skip-save-predictions", action="store_true")
    parser.add_argument("--delete-feature-cache-after-eval", action="store_true")
    parser.add_argument("--eval-size", type=int, default=180)
    parser.add_argument("--T", type=int, default=250)
    parser.add_argument("--sub-t", type=int, default=5)
    parser.add_argument("--sub-x", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--diffusion", type=float, default=1.0)
    parser.add_argument("--noise-scale", type=float, default=0.1)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--max-lag", type=int, default=64)
    parser.add_argument("--crosscorr-block-size", type=int, default=128)
    parser.add_argument("--modes1", type=int, default=16)
    parser.add_argument("--modes2", type=int, default=16)
    parser.add_argument("--modes3", type=int, default=16)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--time-padding", type=int, default=6)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.feature_cache_dir is None:
        args.feature_cache_dir = args.output_dir / "features"

    device = torch.device(args.device)
    model = build_model(args, device)
    load_checkpoint(model, args.checkpoint, device)
    results = []
    for index, data_path in enumerate(args.eval_data_paths, start=1):
        results.append(evaluate_one(args, model, data_path, index, device))
    summary = summarize(results)

    output = {
        "checkpoint": str(args.checkpoint),
        "feature_names": list(FEATURE_NAMES),
        "results": results,
        "summary": summary,
    }
    output_path = args.output_dir / "evaluation_summary.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(f"evaluation summary: {output_path}")


if __name__ == "__main__":
    main()
