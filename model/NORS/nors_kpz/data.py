"""KPZ MAT loading and resumable NORS feature caching."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io
import torch

from .features import FEATURE_NAMES, KPZFeatureBuilder


@dataclass(frozen=True)
class KPZData:
    noise: np.ndarray
    solution: np.ndarray
    time: np.ndarray
    x: np.ndarray
    source_path: Path


def _grid(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).squeeze()


def load_kpz(
    path: str | Path,
    *,
    max_time: int | None = 51,
    sub_t: int = 1,
    sub_x: int = 1,
    spatial_points: int | None = 128,
) -> KPZData:
    """Load KPZ data in ``[sample,time,x]`` order.

    Generated files contain both periodic endpoints. By default the final
    duplicate endpoint is dropped to match the 128-point FNO convention.
    """

    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    raw = {
        key: value
        for key, value in scipy.io.loadmat(path).items()
        if not key.startswith("__")
    }
    missing = {"W", "sol", "T", "X"} - raw.keys()
    if missing:
        raise KeyError(f"{path} is missing required MAT keys: {sorted(missing)}")

    noise = np.asarray(raw["W"], dtype=np.float32)
    solution = np.asarray(raw["sol"], dtype=np.float32)
    time_full = _grid(raw["T"])
    x_full = _grid(raw["X"])
    if noise.ndim != 3 or solution.ndim != 3 or solution.shape != noise.shape:
        raise ValueError(
            f"Expected matching three-dimensional W and sol, got "
            f"W={noise.shape}, sol={solution.shape}."
        )

    if noise.shape[1] == len(time_full):
        pass
    elif noise.shape[2] == len(time_full):
        noise = np.transpose(noise, (0, 2, 1))
        solution = np.transpose(solution, (0, 2, 1))
    else:
        raise ValueError(
            f"Cannot identify time axis of W={noise.shape} from T={time_full.shape}."
        )
    if noise.shape[2] != len(x_full):
        raise ValueError(f"W={noise.shape} is inconsistent with X={x_full.shape}.")

    stop_t = noise.shape[1] if max_time is None else min(int(max_time), noise.shape[1])
    stop_x = noise.shape[2] if spatial_points is None else min(
        int(spatial_points), noise.shape[2]
    )
    noise = noise[:, :stop_t:sub_t, :stop_x:sub_x]
    solution = solution[:, :stop_t:sub_t, :stop_x:sub_x]
    time = time_full[:stop_t:sub_t]
    x = x_full[:stop_x:sub_x]
    expected = (len(time), len(x))
    if tuple(noise.shape[1:]) != expected:
        raise ValueError(
            f"Subsampled W={noise.shape} is inconsistent with grids={expected}."
        )
    return KPZData(
        noise=noise,
        solution=solution,
        time=time,
        x=x,
        source_path=path,
    )


def feature_cache_path(
    cache_dir: str | Path,
    data_path: str | Path,
    *,
    sub_t: int,
    sub_x: int,
    spatial_points: int | None,
    noise_scale: float,
    renorm_constant: float,
) -> Path:
    stem = Path(data_path).stem
    scale = str(noise_scale).replace(".", "p")
    renorm = str(renorm_constant).replace(".", "p")
    points = "all" if spatial_points is None else str(spatial_points)
    name = (
        f"{stem}_nt{sub_t}_nx{sub_x}_points{points}_"
        f"sigma{scale}_C{renorm}_kpz_nors7.npy"
    )
    return Path(cache_dir) / name


def _write_metadata(path: Path, expected: dict, completed_samples: int) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump({**expected, "completed_samples": completed_samples}, handle, indent=2)


def build_feature_cache(
    data: KPZData,
    cache_path: str | Path,
    *,
    device: torch.device,
    batch_size: int,
    diffusion: float,
    noise_scale: float,
    renorm_constant: float,
    rebuild: bool = False,
) -> np.ndarray:
    """Create or resume a ``[N,T,X,7]`` memory-mapped feature cache."""

    cache_path = Path(cache_path)
    metadata_path = cache_path.with_suffix(".json")
    expected_shape = (*data.noise.shape, len(FEATURE_NAMES))
    stat = data.source_path.stat()
    expected_metadata = {
        "shape": list(expected_shape),
        "features": list(FEATURE_NAMES),
        "diffusion": float(diffusion),
        "noise_scale": float(noise_scale),
        "renorm_constant": float(renorm_constant),
        "differentiate_noise": True,
        "source_path": str(data.source_path),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
    }

    completed = 0
    cached = None
    if cache_path.exists() and metadata_path.exists() and not rebuild:
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            matches = all(
                metadata.get(key) == value
                for key, value in expected_metadata.items()
            )
            if matches:
                candidate = np.load(cache_path, mmap_mode="r+")
                if candidate.shape == expected_shape:
                    cached = candidate
                    completed = min(
                        int(metadata.get("completed_samples", expected_shape[0])),
                        expected_shape[0],
                    )
        except (OSError, ValueError, json.JSONDecodeError):
            cached = None

    if cached is None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cached = np.lib.format.open_memmap(
            cache_path,
            mode="w+",
            dtype=np.float32,
            shape=expected_shape,
        )
        _write_metadata(metadata_path, expected_metadata, 0)
    if completed == expected_shape[0]:
        del cached
        return np.load(cache_path, mmap_mode="r")

    builder = KPZFeatureBuilder(
        data.time,
        data.x,
        diffusion=diffusion,
        noise_scale=noise_scale,
        renorm_constant=renorm_constant,
        differentiate_noise=True,
    ).to(device)
    builder.eval()
    with torch.no_grad():
        for start in range(completed, len(data.noise), batch_size):
            stop = min(start + batch_size, len(data.noise))
            noise = torch.from_numpy(
                np.array(data.noise[start:stop], copy=True)
            ).to(device)
            u0 = torch.from_numpy(
                np.array(data.solution[start:stop, 0], copy=True)
            ).to(device)
            cached[start:stop] = builder(noise, u0).cpu().numpy()
            cached.flush()
            _write_metadata(metadata_path, expected_metadata, stop)
            print(f"[kpz features] {stop}/{len(data.noise)}")

    del cached
    return np.load(cache_path, mmap_mode="r")
