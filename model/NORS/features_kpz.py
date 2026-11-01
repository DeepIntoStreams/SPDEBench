"""Derivative regularity-structure features for the one-dimensional KPZ equation."""

from __future__ import annotations

import torch
from torch import nn


FEATURE_NAMES = (
    "I[xi]",
    "I_c[u_0]",
    "I[I[xi]]",
    "I[I_c[u_0]]",
    "I[(D I[xi])^2-C]",
    "I[(D I[xi])(D I_c[u_0])]",
    "I[(D I_c[u_0])^2]",
)


def periodic_heat_matrix(
    n_points: int,
    dt: float,
    dx: float,
    diffusion: float,
) -> torch.Tensor:
    laplacian = torch.diag(-2.0 * torch.ones(n_points))
    laplacian += torch.diag(torch.ones(n_points - 1), diagonal=1)
    laplacian += torch.diag(torch.ones(n_points - 1), diagonal=-1)
    laplacian[0, -1] = 1.0
    laplacian[-1, 0] = 1.0
    identity = torch.eye(n_points)
    return torch.linalg.inv(identity - diffusion * dt * laplacian / (dx**2)).T


class KPZFeatureBuilder(nn.Module):
    """Build width-two, height-two derivative features for KPZ.

    Inputs and outputs use ``[batch,time,x]`` and
    ``[batch,time,x,feature]`` ordering respectively.
    """

    def __init__(
        self,
        time_grid,
        x_grid,
        *,
        diffusion: float = 1.0,
        noise_scale: float = 0.1,
        renorm_constant: float = 128.0,
        differentiate_noise: bool = True,
    ) -> None:
        super().__init__()
        time = torch.as_tensor(time_grid, dtype=torch.float32).flatten()
        x = torch.as_tensor(x_grid, dtype=torch.float32).flatten()
        if min(len(time), len(x)) < 2:
            raise ValueError("The time and space grids must contain at least two points.")

        self.nt = len(time)
        self.nx = len(x)
        self.dt = float(time[1] - time[0])
        self.dx = float(x[1] - x[0])
        self.noise_scale = float(noise_scale)
        self.renorm_constant = float(renorm_constant)
        self.differentiate_noise = bool(differentiate_noise)
        self.register_buffer(
            "heat_matrix",
            periodic_heat_matrix(self.nx, self.dt, self.dx, float(diffusion)),
        )

    def _check_path(self, value: torch.Tensor, name: str) -> None:
        expected = (self.nt, self.nx)
        if value.ndim != 3 or tuple(value.shape[1:]) != expected:
            raise ValueError(
                f"{name} must have shape [batch,{self.nt},{self.nx}], got {tuple(value.shape)}."
            )

    def heat_integrate(self, source: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(source)
        matrix = self.heat_matrix.to(device=source.device, dtype=source.dtype)
        for time_index in range(1, self.nt):
            result[:, time_index] = (
                result[:, time_index - 1] + source[:, time_index] * self.dt
            ) @ matrix
        return result

    def initial_condition_path(self, u0: torch.Tensor) -> torch.Tensor:
        if u0.ndim != 2 or u0.shape[1] != self.nx:
            raise ValueError(f"u0 must have shape [batch,{self.nx}], got {tuple(u0.shape)}.")
        result = torch.zeros(
            u0.shape[0],
            self.nt,
            self.nx,
            dtype=u0.dtype,
            device=u0.device,
        )
        result[:, 0] = u0
        matrix = self.heat_matrix.to(device=u0.device, dtype=u0.dtype)
        for time_index in range(1, self.nt):
            result[:, time_index] = result[:, time_index - 1] @ matrix
        return result

    def spatial_derivative(self, value: torch.Tensor) -> torch.Tensor:
        return (torch.roll(value, -1, dims=-1) - torch.roll(value, 1, dims=-1)) / (
            2.0 * self.dx
        )

    def noise_distribution(self, brownian_field: torch.Tensor) -> torch.Tensor:
        self._check_path(brownian_field, "W")
        if not self.differentiate_noise:
            return brownian_field * self.noise_scale
        xi = torch.zeros_like(brownian_field)
        xi[:, 1:] = (
            torch.diff(brownian_field, dim=1) * (self.noise_scale / self.dt)
        )
        return xi

    def forward(self, brownian_field: torch.Tensor, u0: torch.Tensor) -> torch.Tensor:
        integrated_noise = self.heat_integrate(
            self.noise_distribution(brownian_field)
        )
        initial_path = self.initial_condition_path(u0)
        derivative_noise = self.spatial_derivative(integrated_noise)
        derivative_initial = self.spatial_derivative(initial_path)

        features = (
            integrated_noise,
            initial_path,
            self.heat_integrate(integrated_noise),
            self.heat_integrate(initial_path),
            self.heat_integrate(
                derivative_noise.square() - self.renorm_constant
            ),
            self.heat_integrate(derivative_noise * derivative_initial),
            self.heat_integrate(derivative_initial.square()),
        )
        return torch.stack(features, dim=-1)
