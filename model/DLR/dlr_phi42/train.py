"""Train Phi42 DLR with the exact seven cached base MFVs used by Phi42 NORS."""

from __future__ import annotations

import sys
from pathlib import Path

from model.NORS.nors_phi42 import train as nors_train
from model.NORS.nors_phi42.data import load_phi42

from .model import Phi42DLR


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "model/result/DLR_Phi42"


def _option_supplied(name: str) -> bool:
    return any(argument == name or argument.startswith(f"{name}=") for argument in sys.argv[1:])


def main() -> None:
    """Reuse the NORS training protocol while replacing only the model."""
    args = nors_train.parse_args()
    if not _option_supplied("--output-dir"):
        args.output_dir = DEFAULT_OUTPUT
    for option, name in (
        ("--modes-x", "modes_x"),
        ("--modes-y", "modes_y"),
        ("--modes-t", "modes_t"),
    ):
        if not _option_supplied(option):
            setattr(args, name, 8)
    if not _option_supplied("--width"):
        args.width = 8
    data = load_phi42(
        args.data_path,
        max_time=args.max_time,
        sub_t=args.sub_t,
        sub_x=args.sub_x,
    )
    time_grid = data.time.copy()
    x_grid = data.x.copy()
    y_grid = data.y.copy()
    del data

    def build_dlr(
        *,
        num_features: int,
        modes_x: int,
        modes_y: int,
        modes_t: int,
        width: int,
        depth: int,
        time_padding: int,
    ) -> Phi42DLR:
        return Phi42DLR(
            time_grid,
            x_grid,
            y_grid,
            num_base_features=num_features,
            diffusion=args.diffusion,
            modes_x=modes_x,
            modes_y=modes_y,
            modes_t=modes_t,
            width=width,
            depth=depth,
            time_padding=time_padding,
        )

    # The shared entry point guarantees identical splitting, optimization,
    # early stopping, target/feature caches, and loss semantics.
    nors_train.parse_args = lambda: args
    nors_train.Phi42NORS = build_dlr
    nors_train.main()


if __name__ == "__main__":
    main()
