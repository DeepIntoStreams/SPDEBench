"""Train fixed-feature NORS on a Phi42 dataset."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import hydra
from omegaconf import DictConfig

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.NORS.data_phi42 import (  # noqa: E402
    build_feature_cache,
    build_target_cache,
    feature_cache_path,
    load_phi42,
    target_cache_path,
)
from model.NORS.features_phi42 import FEATURE_NAMES  # noqa: E402
from model.NORS.fixed_feature_train import (  # noqa: E402
    resolve_device,
    set_seed,
    train_cached_features,
)
from model.NORS.models import NORS2D  # noqa: E402


def _optional_path(value) -> Path | None:
    if value is None:
        return None
    return Path(str(value))


@hydra.main(version_base=None, config_path="../config", config_name="nors_phi42")
def main(cfg: DictConfig) -> None:
    set_seed(int(cfg.seed))
    device = resolve_device(cfg.device)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_phi42(
        cfg.data_path,
        max_time=int(cfg.max_time),
        sub_t=int(cfg.sub_t),
        sub_x=int(cfg.sub_x),
    )
    total_requested = int(cfg.ntrain) + int(cfg.nval) + int(cfg.ntest)
    if total_requested > len(data.noise):
        raise ValueError(
            f"Requested {total_requested} samples, but the dataset contains {len(data.noise)}."
        )

    cache_path = _optional_path(cfg.cache_path) or feature_cache_path(
        output_dir / "features",
        cfg.data_path,
        sub_t=int(cfg.sub_t),
        sub_x=int(cfg.sub_x),
        noise_scale=float(cfg.noise_scale),
    )
    targets_path = _optional_path(cfg.target_cache_path) or target_cache_path(cache_path)
    targets = build_target_cache(
        data,
        targets_path,
        batch_size=int(cfg.feature_batch_size),
        rebuild=bool(cfg.rebuild_features),
    )
    data = replace(data, solution=targets)
    features = build_feature_cache(
        data,
        cache_path,
        device=device,
        batch_size=int(cfg.feature_batch_size),
        diffusion=float(cfg.diffusion),
        noise_scale=float(cfg.noise_scale),
        rebuild=bool(cfg.rebuild_features),
    )
    print(f"features={features.shape}, channels={FEATURE_NAMES}")
    print(
        f"target={targets.shape}, feature_cache={cache_path}, target_cache={targets_path}"
    )

    model = NORS2D(
        num_tree=len(FEATURE_NAMES),
        modes_x=int(cfg.modes_x),
        modes_y=int(cfg.modes_y),
        modes_t=int(cfg.modes_t),
        width=int(cfg.width),
        layers=int(cfg.depth),
        padding=int(cfg.time_padding),
    )
    train_cached_features(
        cfg,
        features=features,
        targets=targets,
        model=model,
        feature_names=list(FEATURE_NAMES),
        cache_path=cache_path,
        target_cache_path=targets_path,
        device=device,
    )


if __name__ == "__main__":
    main()
