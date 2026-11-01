"""Data loading and feature caching for the Phi42 NORS baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.io
import torch

from model.NORS.features_phi42 import FEATURE_NAMES, Phi42FeatureBuilder


@dataclass(frozen=True)
class Phi42Data:
    noise: Any
    solution: Any
    time: np.ndarray
    x: np.ndarray
    y: np.ndarray
    source_path: Path
    solution_key: str


class H5ArrayView:
    """Lazy, subsampled view of one ``[N,T,X,Y]`` HDF5 dataset."""

    def __init__(
        self,
        path: Path,
        key: str,
        source_shape: tuple[int, int, int, int],
        *,
        stop_time: int,
        sub_t: int,
        sub_x: int,
    ) -> None:
        self.path = path
        self.key = key
        self.stop_time = stop_time
        self.sub_t = sub_t
        self.sub_x = sub_x
        self.shape = (
            source_shape[0],
            len(range(0, stop_time, sub_t)),
            len(range(0, source_shape[2], sub_x)),
            len(range(0, source_shape[3], sub_x)),
        )
        self._handle: h5py.File | None = None

    def __len__(self) -> int:
        return self.shape[0]

    def _dataset(self):
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle[self.key]

    def __getitem__(self, sample):
        value = self._dataset()[
            sample,
            : self.stop_time : self.sub_t,
            :: self.sub_x,
            :: self.sub_x,
        ]
        return np.asarray(value, dtype=np.float32)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self) -> None:
        self.close()


def _read_mat(path: Path) -> dict[str, np.ndarray]:
    try:
        return {key: value for key, value in scipy.io.loadmat(path).items() if not key.startswith("__")}
    except NotImplementedError:
        result: dict[str, np.ndarray] = {}
        with h5py.File(path, "r") as handle:
            for key in handle.keys():
                value = handle[key][()]
                if value.ndim > 1:
                    value = np.transpose(value, axes=range(value.ndim - 1, -1, -1))
                result[key] = value
        return result


def _grid(value, fallback_size: int) -> np.ndarray:
    if value is None:
        return np.linspace(0.0, 1.0, fallback_size, dtype=np.float32)
    return np.asarray(value, dtype=np.float32).squeeze()


def load_phi42(
    path: str | Path,
    *,
    max_time: int | None = None,
    sub_t: int = 1,
    sub_x: int = 1,
) -> Phi42Data:
    """Load MAT data or create lazy views over generated HDF5 data."""

    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".h5", ".hdf5"}:
        return _load_phi42_h5(path, max_time=max_time, sub_t=sub_t, sub_x=sub_x)
    if "reno" not in path.name.lower():
        raise ValueError(
            f"Expected a renormalized Phi42 dataset (filename containing 'reno'), got {path.name}."
        )

    raw = _read_mat(path)
    missing = {"W", "sol"} - raw.keys()
    if missing:
        raise KeyError(f"{path} is missing required MAT keys: {sorted(missing)}")

    noise = np.asarray(raw["W"], dtype=np.float32)
    solution = np.asarray(raw["sol"], dtype=np.float32)
    if noise.ndim != 4 or solution.ndim != 4:
        raise ValueError(f"Expected four-dimensional W and sol, got {noise.shape}, {solution.shape}.")

    time = _grid(raw.get("T"), noise.shape[1])
    if noise.shape[1] == len(time):
        pass
    elif noise.shape[-1] == len(time):
        noise = np.transpose(noise, (0, 3, 1, 2))
        solution = np.transpose(solution, (0, 3, 1, 2))
    else:
        raise ValueError(f"Cannot identify the time axis of W={noise.shape} using T={time.shape}.")

    stop = noise.shape[1] if max_time is None else min(int(max_time), noise.shape[1])
    noise = noise[:, :stop:sub_t, ::sub_x, ::sub_x]
    solution = solution[:, :stop:sub_t, ::sub_x, ::sub_x]
    time = time[:stop:sub_t]
    x = _grid(raw.get("X"), noise.shape[2] * sub_x)[::sub_x]
    y = _grid(raw.get("Y"), noise.shape[3] * sub_x)[::sub_x]

    expected = (len(time), len(x), len(y))
    if tuple(noise.shape[1:]) != expected or solution.shape != noise.shape:
        raise ValueError(
            f"Inconsistent Phi42 arrays: W={noise.shape}, sol={solution.shape}, grid={expected}."
        )
    return Phi42Data(
        noise=noise,
        solution=solution,
        time=time,
        x=x,
        y=y,
        source_path=path,
        solution_key="sol",
    )


def _load_phi42_h5(
    path: Path,
    *,
    max_time: int | None,
    sub_t: int,
    sub_x: int,
) -> Phi42Data:
    with h5py.File(path, "r") as handle:
        solution_key = "sol_reno" if "sol_reno" in handle else "sol"
        missing = {"W", "X", "Y", "T"} - handle.keys()
        if missing or solution_key not in handle:
            missing_names = sorted(missing | ({"sol_reno or sol"} if solution_key not in handle else set()))
            raise KeyError(f"{path} is missing required HDF5 keys: {missing_names}")
        noise_shape = tuple(int(value) for value in handle["W"].shape)
        solution_shape = tuple(int(value) for value in handle[solution_key].shape)
        time_full = _grid(handle["T"][()], noise_shape[1])
        x_full = _grid(handle["X"][()], noise_shape[2])
        y_full = _grid(handle["Y"][()], noise_shape[3])

    if len(noise_shape) != 4 or solution_shape != noise_shape:
        raise ValueError(
            f"Expected matching [N,T,X,Y] HDF5 arrays, got W={noise_shape}, "
            f"{solution_key}={solution_shape}."
        )
    if noise_shape[1:] != (len(time_full), len(x_full), len(y_full)):
        raise ValueError(
            f"HDF5 arrays use an unsupported axis order: W={noise_shape}, "
            f"grids={(len(time_full), len(x_full), len(y_full))}."
        )

    stop = noise_shape[1] if max_time is None else min(int(max_time), noise_shape[1])
    view_options = {"stop_time": stop, "sub_t": sub_t, "sub_x": sub_x}
    noise = H5ArrayView(path, "W", noise_shape, **view_options)
    solution = H5ArrayView(path, solution_key, solution_shape, **view_options)
    return Phi42Data(
        noise=noise,
        solution=solution,
        time=time_full[:stop:sub_t],
        x=x_full[::sub_x],
        y=y_full[::sub_x],
        source_path=path,
        solution_key=solution_key,
    )


def feature_cache_path(
    cache_dir: str | Path,
    data_path: str | Path,
    *,
    sub_t: int,
    sub_x: int,
    noise_scale: float,
) -> Path:
    stem = Path(data_path).stem
    scale = str(noise_scale).replace(".", "p")
    return Path(cache_dir) / f"{stem}_st{sub_t}_sx{sub_x}_sigma{scale}_nors7.npy"


def target_cache_path(feature_path: str | Path) -> Path:
    feature_path = Path(feature_path)
    return feature_path.with_name(f"{feature_path.stem}_targets.npy")


def _source_metadata(data: Phi42Data) -> dict[str, str | int]:
    stat = data.source_path.stat()
    return {
        "source_path": str(data.source_path),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "solution_key": data.solution_key,
    }


def _write_metadata(path: Path, expected: dict, completed_samples: int) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump({**expected, "completed_samples": completed_samples}, handle, indent=2)


def _open_resumable_cache(
    cache_path: Path,
    expected_shape: tuple[int, ...],
    expected_metadata: dict,
    rebuild: bool,
):
    metadata_path = cache_path.with_suffix(".json")
    if cache_path.exists() and metadata_path.exists() and not rebuild:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        metadata_matches = all(metadata.get(key) == value for key, value in expected_metadata.items())
        if metadata_matches:
            cached = np.load(cache_path, mmap_mode="r+")
            if cached.shape == expected_shape:
                completed = int(metadata.get("completed_samples", expected_shape[0]))
                return cached, min(completed, expected_shape[0]), metadata_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = np.lib.format.open_memmap(
        cache_path,
        mode="w+",
        dtype=np.float32,
        shape=expected_shape,
    )
    _write_metadata(metadata_path, expected_metadata, 0)
    return cached, 0, metadata_path


def build_target_cache(
    data: Phi42Data,
    cache_path: str | Path,
    *,
    batch_size: int,
    rebuild: bool = False,
) -> np.ndarray:
    """Materialize lazy HDF5 targets once as a resumable NumPy memmap."""

    if isinstance(data.solution, np.ndarray):
        return data.solution

    cache_path = Path(cache_path)
    expected_shape = tuple(data.solution.shape)
    expected_metadata = {
        "shape": list(expected_shape),
        "dtype": "float32",
        "kind": "target",
        **_source_metadata(data),
    }
    cached, completed, metadata_path = _open_resumable_cache(
        cache_path, expected_shape, expected_metadata, rebuild
    )
    if completed == expected_shape[0]:
        return np.load(cache_path, mmap_mode="r")

    for start in range(completed, expected_shape[0], batch_size):
        stop = min(start + batch_size, expected_shape[0])
        cached[start:stop] = data.solution[start:stop]
        cached.flush()
        _write_metadata(metadata_path, expected_metadata, stop)
        print(f"[targets] {stop}/{expected_shape[0]}")
    del cached
    return np.load(cache_path, mmap_mode="r")


def build_feature_cache(
    data: Phi42Data,
    cache_path: str | Path,
    *,
    device: torch.device,
    batch_size: int,
    diffusion: float,
    noise_scale: float,
    rebuild: bool = False,
) -> np.ndarray:
    """Create or memory-map the full ``[N,T,X,Y,7]`` NORS feature tensor."""

    cache_path = Path(cache_path)
    expected_shape = (*data.noise.shape, len(FEATURE_NAMES))
    expected_metadata = {
        "shape": list(expected_shape),
        "features": list(FEATURE_NAMES),
        "diffusion": float(diffusion),
        "noise_scale": float(noise_scale),
        "differentiate_noise": True,
        **_source_metadata(data),
    }
    cached, completed, metadata_path = _open_resumable_cache(
        cache_path, expected_shape, expected_metadata, rebuild
    )
    if completed == expected_shape[0]:
        return np.load(cache_path, mmap_mode="r")

    builder = Phi42FeatureBuilder(
        data.time,
        data.x,
        data.y,
        diffusion=diffusion,
        noise_scale=noise_scale,
        differentiate_noise=True,
    ).to(device)
    builder.eval()

    with torch.no_grad():
        for start in range(completed, len(data.noise), batch_size):
            stop = min(start + batch_size, len(data.noise))
            noise = torch.from_numpy(np.asarray(data.noise[start:stop])).to(device)
            solution = np.asarray(data.solution[start:stop])
            u0 = torch.from_numpy(solution[:, 0]).to(device)
            cached[start:stop] = builder(noise, u0).cpu().numpy()
            cached.flush()
            _write_metadata(metadata_path, expected_metadata, stop)
            print(f"[features] {stop}/{len(data.noise)}")

    del cached
    return np.load(cache_path, mmap_mode="r")
