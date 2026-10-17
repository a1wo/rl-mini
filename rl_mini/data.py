"""MNIST loader with no torchvision dependency.

Downloads the four original IDX files once into ``data/`` (about 11 MB) and caches
them. Mirrors are tried in order; the canonical LeCun host now rate-limits.
"""

from __future__ import annotations

import gzip
import os
import struct
import urllib.request
from pathlib import Path

import numpy as np

__all__ = ["load_mnist", "DEFAULT_DATA_DIR"]

DEFAULT_DATA_DIR = Path(
    os.environ.get("RL_MINI_DATA_DIR", Path(__file__).resolve().parent.parent / "data")
)

MIRRORS = (
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
)

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download(name: str, dest: Path) -> None:
    errors = []
    for mirror in MIRRORS:
        url = mirror + name
        try:
            print(f"downloading {url}")
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            with urllib.request.urlopen(url, timeout=60) as response:
                tmp.write_bytes(response.read())
            tmp.replace(dest)
            return
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            errors.append(f"{url}: {exc}")
    raise RuntimeError("could not download MNIST:\n  " + "\n  ".join(errors))


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        zero, dtype_code, n_dims = struct.unpack(">HBB", handle.read(4))
        if zero != 0 or dtype_code != 0x08:
            raise ValueError(f"{path}: unexpected IDX header (not unsigned bytes)")
        dims = struct.unpack(f">{n_dims}I", handle.read(4 * n_dims))
        return np.frombuffer(handle.read(), dtype=np.uint8).reshape(dims)


def load_mnist(
    data_dir: Path | str | None = None,
    *,
    n_train: int | None = None,
    n_test: int | None = None,
) -> dict[str, np.ndarray]:
    """Load MNIST as flat float32 features in [0, 1].

    Args:
        data_dir: cache directory. Defaults to ``./data`` or ``$RL_MINI_DATA_DIR``.
        n_train: keep only the first N training examples.
        n_test: keep only the first N test examples.

    Returns:
        ``{"X_train", "y_train", "X_test", "y_test"}``.
    """
    directory = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    directory.mkdir(parents=True, exist_ok=True)

    raw = {}
    for key, name in FILES.items():
        path = directory / name
        if not path.exists():
            _download(name, path)
        raw[key] = _read_idx(path)

    def images(arr: np.ndarray, limit: int | None) -> np.ndarray:
        arr = arr[:limit] if limit else arr
        return (arr.reshape(arr.shape[0], -1).astype(np.float32)) / 255.0

    def labels(arr: np.ndarray, limit: int | None) -> np.ndarray:
        return (arr[:limit] if limit else arr).astype(np.int64)

    return {
        "X_train": images(raw["train_images"], n_train),
        "y_train": labels(raw["train_labels"], n_train),
        "X_test": images(raw["test_images"], n_test),
        "y_test": labels(raw["test_labels"], n_test),
    }
