"""Three-dimensional Fourier neural operator used by the Phi42 NORS baseline."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SpectralConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_x: int,
        modes_y: int,
        modes_t: int,
    ) -> None:
        super().__init__()
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_t = modes_t
        scale = 1.0 / (in_channels * out_channels)
        shape = (in_channels, out_channels, modes_x, modes_y, modes_t)
        self.weights = nn.ParameterList(
            [nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat)) for _ in range(4)]
        )

    @staticmethod
    def multiply(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixyt,ioxyt->boxyt", inputs, weights)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch = value.shape[0]
        value_ft = torch.fft.rfftn(value, dim=(-3, -2, -1))
        output_ft = torch.zeros(
            batch,
            self.weights[0].shape[1],
            value.shape[-3],
            value.shape[-2],
            value.shape[-1] // 2 + 1,
            dtype=torch.cfloat,
            device=value.device,
        )
        mx = min(self.modes_x, value.shape[-3])
        my = min(self.modes_y, value.shape[-2])
        mt = min(self.modes_t, value_ft.shape[-1])
        slices = (
            (slice(None, mx), slice(None, my)),
            (slice(-mx, None), slice(None, my)),
            (slice(None, mx), slice(-my, None)),
            (slice(-mx, None), slice(-my, None)),
        )
        for weights, (x_slice, y_slice) in zip(self.weights, slices):
            output_ft[:, :, x_slice, y_slice, :mt] = self.multiply(
                value_ft[:, :, x_slice, y_slice, :mt],
                weights[:, :, :mx, :my, :mt],
            )
        return torch.fft.irfftn(output_ft, s=value.shape[-3:])


class FourierBlock3d(nn.Module):
    def __init__(self, modes_x: int, modes_y: int, modes_t: int, width: int, last: bool) -> None:
        super().__init__()
        self.last = last
        self.spectral = SpectralConv3d(width, width, modes_x, modes_y, modes_t)
        self.local = nn.Conv3d(width, width, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.spectral(value) + self.local(value)
        return value if self.last else F.gelu(value)


class Phi42NORS(nn.Module):
    """Map seven full-time RS features to a full Phi42 solution trajectory."""

    def __init__(
        self,
        *,
        num_features: int = 7,
        modes_x: int = 8,
        modes_y: int = 8,
        modes_t: int = 8,
        width: int = 16,
        depth: int = 4,
        time_padding: int = 6,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be positive.")
        self.time_padding = time_padding
        self.lift = nn.Linear(num_features + 3, width)
        self.blocks = nn.Sequential(
            *[
                FourierBlock3d(
                    modes_x,
                    modes_y,
                    modes_t,
                    width,
                    last=layer_index == depth - 1,
                )
                for layer_index in range(depth)
            ]
        )
        self.project = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    @staticmethod
    def coordinate_grid(value: torch.Tensor) -> torch.Tensor:
        batch, nt, nx, ny = value.shape[:4]
        time = torch.linspace(0, 1, nt, device=value.device, dtype=value.dtype)
        x = torch.linspace(0, 1, nx, device=value.device, dtype=value.dtype)
        y = torch.linspace(0, 1, ny, device=value.device, dtype=value.dtype)
        grid_t = time.view(1, nt, 1, 1, 1).expand(batch, nt, nx, ny, 1)
        grid_x = x.view(1, 1, nx, 1, 1).expand(batch, nt, nx, ny, 1)
        grid_y = y.view(1, 1, 1, ny, 1).expand(batch, nt, nx, ny, 1)
        return torch.cat((grid_x, grid_y, grid_t), dim=-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 5:
            raise ValueError(f"Expected [B,T,X,Y,C] features, got {tuple(features.shape)}.")
        value = torch.cat((features, self.coordinate_grid(features)), dim=-1)
        value = self.lift(value.float()).permute(0, 4, 2, 3, 1)
        if self.time_padding:
            value = F.pad(value, (0, self.time_padding))
        value = self.blocks(value)
        if self.time_padding:
            value = value[..., : -self.time_padding]
        value = value.permute(0, 4, 2, 3, 1)
        return self.project(value).squeeze(-1)
