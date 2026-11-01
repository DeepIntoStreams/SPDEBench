"""One-dimensional Fourier backbone for KPZ NORS features."""

from __future__ import annotations

from model.NORS.models import NORS1D


class KPZNORS(NORS1D):
    def __init__(
        self,
        *,
        num_features: int = 7,
        modes_x: int = 16,
        modes_t: int = 16,
        width: int = 32,
        depth: int = 4,
        time_padding: int = 6,
    ) -> None:
        super().__init__(
            num_tree=num_features,
            modes_x=modes_x,
            modes_t=modes_t,
            width=width,
            layers=depth,
            padding=time_padding,
        )
