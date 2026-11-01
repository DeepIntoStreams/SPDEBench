"""Full-time NORS model features for the two-dimensional Phi^4 equation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


FEATURE_NAMES = (
    "I[xi]",
    "I_c[u_0]",
    "I[I[xi]]",
    "I[I_c[u_0]]",
    "I[(I[xi])^2]",
    "I[(I[xi])(I_c[u_0])]",
    "I[(I_c[u_0])^2]",
)


class Phi42FeatureBuilder(nn.Module):
    """Construct the seven width-two, height-two NORS features.

    Inputs use ``[batch, time, x, y]`` ordering.  The default noise convention
    matches SPDEBench Phi42 files: ``W`` is a cumulative Brownian field, so xi
    is ``noise_scale * dW/dt``.  Features are evaluated on the complete time
    grid and returned as ``[batch, time, x, y, feature]``.
    """

    def __init__(
        self,
        time_grid,
        x_grid,
        y_grid,
        *,
        diffusion: float = 1.0,
        noise_scale: float = 0.1,
        differentiate_noise: bool = True,
    ) -> None:
        super().__init__()
        time = torch.as_tensor(time_grid, dtype=torch.float32).flatten()
        x = torch.as_tensor(x_grid, dtype=torch.float32).flatten()
        y = torch.as_tensor(y_grid, dtype=torch.float32).flatten()
        if min(len(time), len(x), len(y)) < 2:
            raise ValueError("The time and spatial grids must each contain at least two points.")

        self.nt, self.nx, self.ny = len(time), len(x), len(y)
        self.dt = float(time[1] - time[0])
        self.dx = float(x[1] - x[0])
        self.dy = float(y[1] - y[0])
        if not torch.isclose(torch.tensor(self.dx), torch.tensor(self.dy), rtol=1e-5, atol=1e-7):
            raise ValueError("The NORS nine-point heat stencil requires dx == dy.")

        self.noise_scale = float(noise_scale)
        self.differentiate_noise = bool(differentiate_noise)

        laplacian = torch.tensor(
            [[[[0.25, 0.5, 0.25], [0.5, -3.0, 0.5], [0.25, 0.5, 0.25]]]],
            dtype=torch.float32,
        )
        identity = torch.zeros_like(laplacian)
        identity[..., 1, 1] = 1.0
        heat_filter = identity + float(diffusion) * self.dt * laplacian / (self.dx**2)
        self.register_buffer("heat_filter", heat_filter)

    def _check_shape(self, value: torch.Tensor, name: str) -> None:
        expected = (self.nt, self.nx, self.ny)
        if value.ndim != 4 or tuple(value.shape[1:]) != expected:
            raise ValueError(
                f"{name} must have shape [batch, {self.nt}, {self.nx}, {self.ny}], "
                f"got {tuple(value.shape)}."
            )

    def heat_step(self, value: torch.Tensor) -> torch.Tensor:
        filt = self.heat_filter.to(device=value.device, dtype=value.dtype)
        padded = F.pad(value.unsqueeze(1), (1, 1, 1, 1), mode="circular")
        return F.conv2d(padded, filt).squeeze(1)

    def heat_integrate(self, source: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(source)
        for time_index in range(1, self.nt):
            result[:, time_index] = (
                self.heat_step(result[:, time_index - 1])
                + source[:, time_index] * self.dt
            )
        return result

    def initial_condition_path(self, u0: torch.Tensor) -> torch.Tensor:
        if u0.ndim != 3 or tuple(u0.shape[1:]) != (self.nx, self.ny):
            raise ValueError(
                f"u0 must have shape [batch, {self.nx}, {self.ny}], got {tuple(u0.shape)}."
            )
        result = torch.zeros(
            u0.shape[0],
            self.nt,
            self.nx,
            self.ny,
            dtype=u0.dtype,
            device=u0.device,
        )
        result[:, 0] = u0
        for time_index in range(1, self.nt):
            result[:, time_index] = self.heat_step(result[:, time_index - 1])
        return result

    def noise_distribution(self, brownian_field: torch.Tensor) -> torch.Tensor:
        self._check_shape(brownian_field, "W")
        if not self.differentiate_noise:
            return brownian_field * self.noise_scale
        xi = torch.zeros_like(brownian_field)
        xi[:, 1:] = torch.diff(brownian_field, dim=1) * (self.noise_scale / self.dt)
        return xi

    def forward(self, brownian_field: torch.Tensor, u0: torch.Tensor) -> torch.Tensor:
        xi = self.noise_distribution(brownian_field)
        integrated_noise = self.heat_integrate(xi)
        initial_path = self.initial_condition_path(u0)

        features = (
            integrated_noise,
            initial_path,
            self.heat_integrate(integrated_noise),
            self.heat_integrate(initial_path),
            self.heat_integrate(integrated_noise.square()),
            self.heat_integrate(integrated_noise * initial_path),
            self.heat_integrate(initial_path.square()),
        )
        return torch.stack(features, dim=-1)
