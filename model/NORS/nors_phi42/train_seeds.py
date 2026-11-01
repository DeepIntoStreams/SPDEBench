"""Run Phi42 NORS training sequentially for multiple seeds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from .data import feature_cache_path


SUMMARY_KEYS = [
    "train_relative_l2",
    "validation_relative_l2",
    "test_relative_l2",
    "best_validation_relative_l2",
    "training_seconds",
    "completed_epoch",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3407, 3408, 3409, 3410, 3411])
    parser.add_argument(
        "--checkpoint-template",
        default="NORS_Phi42+_reno_xi_eps128_1200_best_val_seed{seed}.pt",
    )
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--max-time", type=int, default=250)
    parser.add_argument("--sub-t", type=int, default=5)
    parser.add_argument("--sub-x", type=int, default=1)
    parser.add_argument("--diffusion", type=float, default=1.0)
    parser.add_argument("--noise-scale", type=float, default=0.1)
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--ntrain", type=int, default=840)
    parser.add_argument("--nval", type=int, default=180)
    parser.add_argument("--ntest", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--modes-x", type=int, default=16)
    parser.add_argument("--modes-y", type=int, default=16)
    parser.add_argument("--modes-t", type=int, default=16)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--time-padding", type=int, default=6)
    parser.add_argument("--scheduler-patience", type=int, default=50)
    parser.add_argument("--scheduler-factor", type=float, default=0.1)
    parser.add_argument("--scheduler-threshold", type=float, default=1e-6)
    parser.add_argument("--early-stopping", type=int, default=80)
    parser.add_argument("--delta", type=float, default=2e-4)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def append_option(command: list[str], name: str, value) -> None:
    command.extend((f"--{name}", str(value)))


def training_command(args, seed: int, cache_path: Path, seed_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "model.NORS.nors_phi42.train",
        "--data-path",
        str(args.data_path),
        "--output-dir",
        str(seed_dir),
        "--cache-path",
        str(cache_path),
        "--checkpoint-name",
        args.checkpoint_template.format(seed=seed),
        "--seed",
        str(seed),
    ]
    options = {
        "max-time": args.max_time,
        "sub-t": args.sub_t,
        "sub-x": args.sub_x,
        "diffusion": args.diffusion,
        "noise-scale": args.noise_scale,
        "feature-batch-size": args.feature_batch_size,
        "ntrain": args.ntrain,
        "nval": args.nval,
        "ntest": args.ntest,
        "batch-size": args.batch_size,
        "num-workers": args.num_workers,
        "epochs": args.epochs,
        "learning-rate": args.learning_rate,
        "weight-decay": args.weight_decay,
        "modes-x": args.modes_x,
        "modes-y": args.modes_y,
        "modes-t": args.modes_t,
        "width": args.width,
        "depth": args.depth,
        "time-padding": args.time_padding,
        "scheduler-patience": args.scheduler_patience,
        "scheduler-factor": args.scheduler_factor,
        "scheduler-threshold": args.scheduler_threshold,
        "early-stopping": args.early_stopping,
        "delta": args.delta,
        "print-every": args.print_every,
        "device": args.device,
    }
    for name, value in options.items():
        append_option(command, name, value)
    return command


def summarize(rows):
    summary = {}
    print("\nTraining seed summary (mean +/- std):")
    for key in SUMMARY_KEYS:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary[key] = {"mean": mean, "std": std}
        print(f"{key}: {mean:.6f} +/- {std:.6f}")
    return summary


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.cache_path or feature_cache_path(
        args.output_dir / "shared_features",
        args.data_path,
        sub_t=args.sub_t,
        sub_x=args.sub_x,
        noise_scale=args.noise_scale,
    )

    rows = []
    for run_index, seed in enumerate(args.seeds):
        seed_dir = args.output_dir / f"seed_{seed}"
        metrics_path = seed_dir / "metrics.json"
        if metrics_path.exists() and not args.rerun:
            print(f"[seed {seed}] reusing {metrics_path}")
        else:
            seed_dir.mkdir(parents=True, exist_ok=True)
            command = training_command(args, seed, cache_path, seed_dir)
            if args.rebuild_features and run_index == 0:
                command.append("--rebuild-features")
            print(f"[seed {seed}] {' '.join(command)}", flush=True)
            subprocess.run(command, check=True)
        with metrics_path.open("r", encoding="utf-8") as handle:
            row = json.load(handle)
        row["seed"] = seed
        rows.append(row)

    summary = summarize(rows)
    output = {
        "seeds": args.seeds,
        "shared_feature_cache": str(cache_path),
        "runs": [
            {
                "seed": row["seed"],
                **{key: row[key] for key in SUMMARY_KEYS},
                "stop_reason": row.get("stop_reason"),
            }
            for row in rows
        ],
        "summary": summary,
    }
    output_path = args.output_dir / "training_seed_summary.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(f"training summary: {output_path}")


if __name__ == "__main__":
    main()
