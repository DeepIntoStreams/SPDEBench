"""Evaluate KPZ DLR with exactly the same data, MFV, metric, and timing logic as NORS."""

from __future__ import annotations

import sys

import numpy as np

from model.NORS.nors_kpz import eval as nors_eval
from model.NORS.nors_kpz.data import load_kpz

from .model import KPZDLR


def _option_supplied(name: str) -> bool:
    return any(argument == name or argument.startswith(f"{name}=") for argument in sys.argv[1:])


def main() -> None:
    args = nors_eval.parse_args()
    if not _option_supplied("--width"):
        args.width = 8
    reference = load_kpz(
        args.eval_data_paths[0],
        max_time=args.max_time,
        sub_t=args.sub_t,
        sub_x=args.sub_x,
        spatial_points=args.spatial_points,
    )
    time_grid = reference.time.copy()
    x_grid = reference.x.copy()
    del reference

    def build_dlr(model_args, device):
        model = KPZDLR(
            time_grid,
            x_grid,
            num_base_features=len(nors_eval.FEATURE_NAMES),
            diffusion=model_args.diffusion,
            modes_x=model_args.modes_x,
            modes_t=model_args.modes_t,
            width=model_args.width,
            depth=model_args.depth,
            time_padding=model_args.time_padding,
        ).to(device)
        print(f"The model has {nors_eval.count_parameters(model)} parameters")
        return model

    original_make_loader = nors_eval.make_loader

    def checked_make_loader(data_path, loader_args, device):
        result = original_make_loader(data_path, loader_args, device)
        data = result[0]
        if not (
            np.array_equal(data.time, time_grid)
            and np.array_equal(data.x, x_grid)
        ):
            raise ValueError(
                f"Evaluation grid in {data_path} differs from the first dataset."
            )
        return result

    # Reusing the NORS evaluator makes feature timing, first-batch exclusion,
    # synchronization, per-sample normalization, metrics, and summaries identical.
    nors_eval.parse_args = lambda: args
    nors_eval.build_model = build_dlr
    nors_eval.make_loader = checked_make_loader
    nors_eval.main()


if __name__ == "__main__":
    main()
