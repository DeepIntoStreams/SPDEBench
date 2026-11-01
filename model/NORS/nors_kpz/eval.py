"""Evaluate a KPZ NORS checkpoint on independent MAT datasets."""

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

from .data import build_feature_cache, feature_cache_path, load_kpz
from .features import FEATURE_NAMES
from .model import KPZNORS
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


def make_loader(data_path, args, device):
    data = load_kpz(
        data_path,
        max_time=args.max_time,
        sub_t=args.sub_t,
        sub_x=args.sub_x,
        spatial_points=args.spatial_points,
    )
    if args.eval_size > len(data.noise):
        raise ValueError(
            f"Requested {args.eval_size} samples, dataset has {len(data.noise)}."
        )
    if args.eval_size < len(data.noise):
        start = len(data.noise) - args.eval_size
        data = type(data)(
            noise=data.noise[start:],
            solution=data.solution[start:],
            time=data.time,
            x=data.x,
            source_path=data.source_path,
        )
    cache_path = feature_cache_path(
        args.feature_cache_dir,
        data_path,
        sub_t=args.sub_t,
        sub_x=args.sub_x,
        spatial_points=args.spatial_points,
        noise_scale=args.noise_scale,
        renorm_constant=args.renorm_constant,
    )
    cache_hit = cache_path.exists() and cache_path.with_suffix(".json").exists()
    start_time = default_timer()
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
    feature_seconds = default_timer() - start_time
    loader = DataLoader(
        EvaluationDataset(features, data.solution),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    return data, loader, cache_path, feature_seconds, cache_hit


def evaluate_relative_l2(model, loader, device) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for features, target in loader:
            prediction = model(features.to(device))
            target = target.to(device)
            total += float(relative_l2_sum(prediction[:, 1:], target[:, 1:]))
    return total / len(loader.dataset)


def collect_predictions(model, loader, device):
    model.eval()
    real, prediction = [], []
    with torch.no_grad():
        for features, target in loader:
            prediction.append(model(features.to(device)).detach().cpu())
            real.append(target)
    return torch.cat(real), torch.cat(prediction)


def stable_autocorr(value, max_lag=50, eps=1e-8):
    max_lag = min(max_lag, value.shape[1])
    value = value - value.mean((0, 1), keepdim=True)
    variance = torch.var(value, unbiased=False, dim=(0, 1))
    correlations = []
    for lag in range(max_lag):
        product = value[:, lag:] * value[:, :-lag] if lag else value.square()
        correlation = torch.mean(product, dim=(0, 1)) / (variance + eps)
        correlations.append(
            torch.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
        )
    return torch.stack(correlations)


def evaluate_metrics(model, loader, device, prefix, max_lag=50):
    real_path, prediction_path = collect_predictions(model, loader, device)
    real_field = real_path.permute(0, 2, 1)
    prediction_field = prediction_path.permute(0, 2, 1)
    metric_specs = [
        ("Rel_L2", lambda: LpLossMetric(mode="rel").measure(real_field, prediction_field)),
        ("W1_2", lambda: HsLossMetric(k=1).measure(real_field, prediction_field)),
        ("SigW1", lambda: SigW1Metric().measure(real_field, prediction_field)),
        (
            "AutoCorr",
            lambda: (
                stable_autocorr(prediction_path, max_lag)
                - stable_autocorr(real_path, max_lag)
            )
            .square()
            .mean()
            .sqrt(),
        ),
        (
            "CrossCorr",
            lambda: CrossCorrelationMetric(
                lags=min(max_lag, real_path.shape[1])
            ).measure(real_path, prediction_path),
        ),
    ]
    print(f"{prefix} metrics:", flush=True)
    results = {}
    for name, compute in metric_specs:
        try:
            print(f"computing {prefix}_{name}...", flush=True)
            results[name] = float(compute())
            print(f"{prefix}_{name}: {results[name]:.6f}", flush=True)
        except Exception as error:
            results[name] = None
            print(f"{prefix}_{name}: failed ({error})", flush=True)
    return results, real_field, prediction_field


def measure_inference_time(model, loader, device, warmup_batches=1):
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
    return {
        "avg_sample_time": total_time / total_samples
        if total_samples
        else float("nan"),
        "throughput": total_samples / total_time
        if total_time > 0
        else float("nan"),
        "avg_batch_time": total_time / total_batches
        if total_batches
        else float("nan"),
    }


def build_model(args, device):
    model = KPZNORS(
        num_features=len(FEATURE_NAMES),
        modes_x=args.modes_x,
        modes_t=args.modes_t,
        width=args.width,
        depth=args.depth,
        time_padding=args.time_padding,
    ).to(device)
    print(f"The model has {count_parameters(model)} parameters")
    return model


def evaluate_one(args, model, data_path, index, device):
    data, loader, cache_path, feature_seconds, cache_hit = make_loader(
        data_path, args, device
    )
    print("\n" + "=" * 80)
    print(f"Evaluation dataset {index}: {data_path}")
    print(f"W shape: {data.noise.shape}; sol shape: {data.solution.shape}")
    loss_test = evaluate_relative_l2(model, loader, device)
    inference = measure_inference_time(
        model, loader, device, warmup_batches=args.warmup_batches
    )
    metrics, real, prediction = evaluate_metrics(
        model,
        loader,
        device,
        prefix=f"test_dataset_{index}",
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

    feature_average = feature_seconds / len(loader.dataset)
    inference_average = inference["avg_sample_time"]
    result = {
        "dataset": str(data_path),
        "num_samples": len(loader.dataset),
        "feature_cache": str(cache_path),
        "feature_cache_hit": cache_hit,
        "feature_cache_deleted": False,
        "prediction_file": str(prediction_path) if prediction_path else None,
        "loss_test": loss_test,
        "test_Rel_L2": metrics.get("Rel_L2"),
        "test_W1_2": metrics.get("W1_2"),
        "test_SigW1": metrics.get("SigW1"),
        "test_AutoCorr": metrics.get("AutoCorr"),
        "test_CrossCorr": metrics.get("CrossCorr"),
        "inference_avg_sample_time": inference_average,
        "inference_throughput": inference["throughput"],
        "feature_avg_sample_time": feature_average,
        "end_to_end_avg_sample_time": feature_average + inference_average,
    }
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
    print("\nIndependent test dataset summary (mean +/- std):")
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-data-paths", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path)
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--skip-save-predictions", action="store_true")
    parser.add_argument("--delete-feature-cache-after-eval", action="store_true")
    parser.add_argument("--eval-size", type=int, default=180)
    parser.add_argument("--max-time", type=int, default=51)
    parser.add_argument("--sub-t", type=int, default=1)
    parser.add_argument("--sub-x", type=int, default=1)
    parser.add_argument("--spatial-points", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--diffusion", type=float, default=1.0)
    parser.add_argument("--noise-scale", type=float, default=0.1)
    parser.add_argument("--renorm-constant", type=float, default=128.0)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--max-lag", type=int, default=50)
    parser.add_argument("--modes-x", type=int, default=16)
    parser.add_argument("--modes-t", type=int, default=16)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--time-padding", type=int, default=6)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.feature_cache_dir is None:
        args.feature_cache_dir = args.output_dir / "features"
    device = torch.device(args.device)
    model = build_model(args, device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"loaded checkpoint: {args.checkpoint}")
    results = [
        evaluate_one(args, model, path, index, device)
        for index, path in enumerate(args.eval_data_paths, start=1)
    ]
    output = {
        "checkpoint": str(args.checkpoint),
        "feature_names": list(FEATURE_NAMES),
        "results": results,
        "summary": summarize(results),
    }
    output_path = args.output_dir / "evaluation_summary.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(f"evaluation summary: {output_path}")


if __name__ == "__main__":
    main()
