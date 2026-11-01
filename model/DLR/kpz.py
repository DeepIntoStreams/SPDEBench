"""DLR model with the same seven base MFVs as the KPZ NORS baseline."""

from __future__ import annotations

import torch
from torch import nn

from model.NORS.features_kpz import KPZFeatureBuilder
from model.NORS.models import NORS1D


class KPZLatentFeatureBuilder(nn.Module):
    """Rebuild the four initial-condition-dependent trees from a latent field."""

    def __init__(self, time_grid, x_grid, *, diffusion: float = 1.0) -> None:
        super().__init__()
        self.operator = KPZFeatureBuilder(
            time_grid,
            x_grid,
            diffusion=diffusion,
            noise_scale=1.0,
            renorm_constant=0.0,
        )
        # This deterministic operator belongs to the evaluation grid, not the
        # learned checkpoint. Reconstruct it instead of loading stale spacing.
        self.operator.register_buffer(
            "heat_matrix", self.operator.heat_matrix, persistent=False
        )

    def heat_integrate(self, source: torch.Tensor) -> torch.Tensor:
        """Autograd-safe counterpart of the cached base-feature recurrence."""
        matrix = self.operator.heat_matrix.to(
            device=source.device, dtype=source.dtype
        )
        states = [torch.zeros_like(source[:, 0])]
        for time_index in range(1, self.operator.nt):
            states.append(
                (states[-1] + source[:, time_index] * self.operator.dt) @ matrix
            )
        return torch.stack(states, dim=1)

    def forward(
        self, integrated_noise: torch.Tensor, latent: torch.Tensor
    ) -> torch.Tensor:
        derivative_noise = self.operator.spatial_derivative(integrated_noise)
        derivative_latent = self.operator.spatial_derivative(latent)
        return torch.stack(
            (
                latent,
                self.heat_integrate(latent),
                self.heat_integrate(derivative_noise * derivative_latent),
                self.heat_integrate(derivative_latent.square()),
            ),
            dim=-1,
        )


class KPZDLR(nn.Module):
    """DLR: seven fixed base MFVs, learned latent field, then latent RS features."""

    def __init__(
        self,
        time_grid,
        x_grid,
        *,
        num_base_features: int = 7,
        diffusion: float = 1.0,
        modes_x: int = 16,
        modes_t: int = 16,
        width: int = 8,
        depth: int = 4,
        time_padding: int = 6,
        latent_width: int = 32,
    ) -> None:
        super().__init__()
        if num_base_features != 7:
            raise ValueError("KPZ DLR requires the same seven base MFVs as NORS.")
        self.latent_encoder = nn.Sequential(
            nn.Linear(num_base_features, latent_width),
            nn.GELU(),
            nn.Linear(latent_width, 1),
        )
        self.latent_features = KPZLatentFeatureBuilder(
            time_grid, x_grid, diffusion=diffusion
        )
        self.backbone = NORS1D(
            num_tree=num_base_features + 4,
            modes_x=modes_x,
            modes_t=modes_t,
            width=width,
            layers=depth,
            padding=time_padding,
        )

    def forward(self, base_features: torch.Tensor) -> torch.Tensor:
        if base_features.ndim != 4 or base_features.shape[-1] != 7:
            raise ValueError(
                "Expected KPZ base MFVs with shape [B,T,X,7], "
                f"got {tuple(base_features.shape)}."
            )
        latent = self.latent_encoder(base_features).squeeze(-1)
        rebuilt = self.latent_features(base_features[..., 0], latent)
        return self.backbone(torch.cat((base_features, rebuilt), dim=-1))
