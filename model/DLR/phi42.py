"""DLR model with the same seven base MFVs as the Phi42 NORS baseline."""

from __future__ import annotations

import torch
from torch import nn

from model.NORS.features_phi42 import Phi42FeatureBuilder
from model.NORS.models import NORS2D


class Phi42LatentFeatureBuilder(nn.Module):
    """Rebuild the four initial-condition-dependent trees from a latent field."""

    def __init__(
        self,
        time_grid,
        x_grid,
        y_grid,
        *,
        diffusion: float = 1.0,
    ) -> None:
        super().__init__()
        self.operator = Phi42FeatureBuilder(
            time_grid,
            x_grid,
            y_grid,
            diffusion=diffusion,
            noise_scale=1.0,
        )
        # This deterministic operator belongs to the evaluation grid, not the
        # learned checkpoint. Reconstruct it instead of loading stale spacing.
        self.operator.register_buffer(
            "heat_filter", self.operator.heat_filter, persistent=False
        )

    def heat_integrate(self, source: torch.Tensor) -> torch.Tensor:
        """Autograd-safe counterpart of the cached base-feature recurrence."""
        states = [torch.zeros_like(source[:, 0])]
        for time_index in range(1, self.operator.nt):
            states.append(
                self.operator.heat_step(states[-1])
                + source[:, time_index] * self.operator.dt
            )
        return torch.stack(states, dim=1)

    def forward(
        self, integrated_noise: torch.Tensor, latent: torch.Tensor
    ) -> torch.Tensor:
        return torch.stack(
            (
                latent,
                self.heat_integrate(latent),
                self.heat_integrate(integrated_noise * latent),
                self.heat_integrate(latent.square()),
            ),
            dim=-1,
        )


class Phi42DLR(nn.Module):
    """DLR: seven fixed base MFVs, learned latent field, then latent RS features."""

    def __init__(
        self,
        time_grid,
        x_grid,
        y_grid,
        *,
        num_base_features: int = 7,
        diffusion: float = 1.0,
        modes_x: int = 8,
        modes_y: int = 8,
        modes_t: int = 8,
        width: int = 8,
        depth: int = 4,
        time_padding: int = 6,
        latent_width: int = 32,
    ) -> None:
        super().__init__()
        if num_base_features != 7:
            raise ValueError("Phi42 DLR requires the same seven base MFVs as NORS.")
        self.latent_encoder = nn.Sequential(
            nn.Linear(num_base_features, latent_width),
            nn.ReLU(),
            nn.Linear(latent_width, 1),
        )
        self.latent_features = Phi42LatentFeatureBuilder(
            time_grid,
            x_grid,
            y_grid,
            diffusion=diffusion,
        )
        self.backbone = NORS2D(
            num_tree=num_base_features + 4,
            modes_x=modes_x,
            modes_y=modes_y,
            modes_t=modes_t,
            width=width,
            layers=depth,
            padding=time_padding,
        )
        # NORS2D's decoder is GELU. Phi42 DLR keeps the ReLU head used in the
        # recorded runs.
        self.backbone.decoder = nn.Sequential(
            nn.Linear(width, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, base_features: torch.Tensor) -> torch.Tensor:
        if base_features.ndim != 5 or base_features.shape[-1] != 7:
            raise ValueError(
                "Expected Phi42 base MFVs with shape [B,T,X,Y,7], "
                f"got {tuple(base_features.shape)}."
            )
        latent = self.latent_encoder(base_features).squeeze(-1)
        rebuilt = self.latent_features(base_features[..., 0], latent)
        return self.backbone(torch.cat((base_features, rebuilt), dim=-1))
