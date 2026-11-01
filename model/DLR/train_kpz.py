"""Train KPZ DLR on the same seven cached base MFVs as KPZ NORS."""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.DLR.kpz import KPZDLR  # noqa: E402
from model.NORS.data_kpz import (  # noqa: E402
    build_feature_cache,
    feature_cache_path,
    load_kpz,
)
from model.NORS.features_kpz import FEATURE_NAMES  # noqa: E402
from model.NORS.fixed_feature_train import (  # noqa: E402
    resolve_device,
    set_seed,
    train_cached_features,
)


def _optional_path(value) -> Path | None:
    if value is None:
        return None
    return Path(str(value))


@hydra.main(version_base=None, config_path="../config", config_name="dlr_kpz")
def main(cfg: DictConfig) -> None:
    set_seed(int(cfg.seed))
    device = resolve_device(cfg.device)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_kpz(
        cfg.data_path,
        max_time=int(cfg.max_time),
        sub_t=int(cfg.sub_t),
        sub_x=int(cfg.sub_x),
        spatial_points=int(cfg.spatial_points),
    )
    total_requested = int(cfg.ntrain) + int(cfg.nval) + int(cfg.ntest)
    if total_requested > len(data.noise):
        raise ValueError(
            f"Requested {total_requested} samples, but dataset has {len(data.noise)}."
        )
    time_grid = data.time.copy()
    x_grid = data.x.copy()

    cache_path = _optional_path(cfg.cache_path) or feature_cache_path(
        output_dir / "features",
        cfg.data_path,
        sub_t=int(cfg.sub_t),
        sub_x=int(cfg.sub_x),
        spatial_points=int(cfg.spatial_points),
        noise_scale=float(cfg.noise_scale),
        renorm_constant=float(cfg.renorm_constant),
    )
    features = build_feature_cache(
        data,
        cache_path,
        device=device,
        batch_size=int(cfg.feature_batch_size),
        diffusion=float(cfg.diffusion),
        noise_scale=float(cfg.noise_scale),
        renorm_constant=float(cfg.renorm_constant),
        rebuild=bool(cfg.rebuild_features),
    )
    print(f"features={features.shape}, channels={FEATURE_NAMES}")
    print(f"target={data.solution.shape}, cache={cache_path}")

    model = KPZDLR(
        time_grid,
        x_grid,
        num_base_features=len(FEATURE_NAMES),
        diffusion=float(cfg.diffusion),
        modes_x=int(cfg.modes_x),
        modes_t=int(cfg.modes_t),
        width=int(cfg.width),
        depth=int(cfg.depth),
        time_padding=int(cfg.time_padding),
        latent_width=int(cfg.latent_width),
    )
    train_cached_features(
        cfg,
        features=features,
        targets=data.solution,
        model=model,
        feature_names=list(FEATURE_NAMES),
        cache_path=cache_path,
        target_cache_path=None,
        device=device,
    )


if __name__ == "__main__":
    main()
