"""Evaluate multiple Phi42 NORS seeds and produce the paper-table row."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


TABLE_METRICS = {
    "Rel_L2": ("test_Rel_L2", 1.0),
    "W1_2": ("test_W1_2", 1.0),
    "Corr": ("test_CrossCorr", 1.0),
    "AutoCorr": ("test_AutoCorr", 1.0),
    "Sig_W1": ("test_SigW1", 1.0),
    "Time_ms": ("inference_avg_sample_time", 1000.0),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--eval-data-paths", type=Path, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3407, 3408, 3409, 3410, 3411])
    parser.add_argument(
        "--checkpoint-template",
        default="NORS_Phi42+_reno_xi_eps128_1200_best_val_seed{seed}.pt",
    )
    parser.add_argument("--feature-cache-dir", type=Path)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
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
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def append_option(command: list[str], name: str, value) -> None:
    command.extend((f"--{name}", str(value)))


def evaluation_command(args, checkpoint: Path, output_dir: Path, cache_dir: Path):
    command = [
        sys.executable,
        "-m",
        "model.NORS.nors_phi42.eval",
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        "--feature-cache-dir",
        str(cache_dir),
        "--eval-data-paths",
        *(str(path) for path in args.eval_data_paths),
    ]
    options = {
        "eval-size": args.eval_size,
        "T": args.T,
        "sub-t": args.sub_t,
        "sub-x": args.sub_x,
        "batch-size": args.batch_size,
        "feature-batch-size": args.feature_batch_size,
        "num-workers": args.num_workers,
        "diffusion": args.diffusion,
        "noise-scale": args.noise_scale,
        "warmup-batches": args.warmup_batches,
        "max-lag": args.max_lag,
        "crosscorr-block-size": args.crosscorr_block_size,
        "modes1": args.modes1,
        "modes2": args.modes2,
        "modes3": args.modes3,
        "width": args.width,
        "layers": args.layers,
        "time-padding": args.time_padding,
        "device": args.device,
    }
    for name, value in options.items():
        append_option(command, name, value)
    if not args.save_predictions:
        command.append("--skip-save-predictions")
    return command


def aggregate(seed_summaries):
    table = {}
    print("\nPhi42 NORS table row (mean +/- std across trained seeds):")
    for display_name, (source_name, scale) in TABLE_METRICS.items():
        values = np.asarray(
            [
                row["summary"][source_name]["mean"] * scale
                for row in seed_summaries
                if source_name in row["summary"]
            ],
            dtype=np.float64,
        )
        if values.size == 0:
            continue
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if values.size > 1 else 0.0
        table[display_name] = {"mean": mean, "std": std, "values": values.tolist()}
        print(f"{display_name}: {mean:.6f} +/- {std:.6f}")
    return table


def main():
    args = parse_args()
    args.training_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.feature_cache_dir or args.training_dir / "evaluation_features"
    cache_dir.mkdir(parents=True, exist_ok=True)

    seed_summaries = []
    for run_index, seed in enumerate(args.seeds):
        seed_dir = args.training_dir / f"seed_{seed}"
        checkpoint = seed_dir / args.checkpoint_template.format(seed=seed)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint for seed {seed}: {checkpoint}")
        output_dir = seed_dir / "evaluation"
        summary_path = output_dir / "evaluation_summary.json"
        if summary_path.exists() and not args.rerun:
            print(f"[seed {seed}] reusing {summary_path}")
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            command = evaluation_command(args, checkpoint, output_dir, cache_dir)
            if args.rebuild_features and run_index == 0:
                command.append("--rebuild-features")
            print(f"[seed {seed}] {' '.join(command)}", flush=True)
            subprocess.run(command, check=True)
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        summary["seed"] = seed
        seed_summaries.append(summary)

    table = aggregate(seed_summaries)
    output = {
        "aggregation": (
            "For each training seed, metrics are averaged across independent evaluation "
            "datasets; reported mean and sample std are then computed across training seeds."
        ),
        "seeds": args.seeds,
        "evaluation_datasets": [str(path) for path in args.eval_data_paths],
        "seed_dataset_summaries": [
            {"seed": row["seed"], "summary": row["summary"]} for row in seed_summaries
        ],
        "table": table,
    }
    output_path = args.training_dir / "table18_nors_summary.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(f"table summary: {output_path}")


if __name__ == "__main__":
    main()
