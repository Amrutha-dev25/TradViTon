"""Image loading and preprocessing shared by training, evaluation and prediction.

Pipeline for every image (training AND external images use exactly this):
    read -> (BGR -> RGB) -> resize to 128x128 -> scale to [0, 1] -> (128, 128, 3)

We use cv2.resize (real interpolation) and NEVER numpy.reshape, because source
images have different resolutions; reshape only re-arranges existing numbers
and cannot turn e.g. a 900x600x3 photo into 128x128x3.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PIL import Image, ImageOps

from src import config


def is_supported_image(path: Path) -> bool:
    return path.suffix.lower() in config.SUPPORTED_EXTENSIONS


def discover_images(folder: Path) -> list[Path]:
    """Recursively list supported image files in ``folder`` (sorted)."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.rglob("*") if p.is_file() and is_supported_image(p))


def load_image_rgb(path: str | Path) -> np.ndarray:
    """Read an image as an RGB uint8 array of its ORIGINAL size.

    Uses np.fromfile + cv2.imdecode so that Windows paths with non-ASCII
    characters work. Falls back to Pillow for formats OpenCV cannot decode.
    Raises ValueError if the file is unreadable/corrupted.
    """
    path = Path(path)
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"cannot read file: {exc}") from exc
    if data.size == 0:
        raise ValueError("file is empty")

    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)   # OpenCV returns BGR
    if bgr is not None:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # BGR -> RGB

    try:  # fallback (e.g. some WEBP variants)
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            return np.array(im.convert("RGB"), dtype=np.uint8)
    except Exception as exc:  # noqa: BLE001 - any decode problem means "corrupted"
        raise ValueError(f"cannot decode image: {exc}") from exc


def resize_image(rgb: np.ndarray) -> np.ndarray:
    """Resize (not reshape!) to IMAGE_SIZE using proper interpolation."""
    height, width = config.IMAGE_SIZE
    shrinking = rgb.shape[0] * rgb.shape[1] > height * width
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(rgb, (width, height), interpolation=interpolation)  # (w, h)


def normalize_image(rgb_uint8: np.ndarray) -> np.ndarray:
    """Scale pixel values from [0, 255] to [0, 1] (float32)."""
    return rgb_uint8.astype(np.float32) / 255.0


def preprocess_array(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 array of any size -> float32 array of shape (128, 128, 3)."""
    out = normalize_image(resize_image(rgb))
    if out.shape != config.INPUT_SHAPE:  # safety net, should never trigger
        raise ValueError(f"unexpected shape {out.shape}, expected {config.INPUT_SHAPE}")
    return out


def load_and_preprocess(path: str | Path) -> tuple[np.ndarray, tuple[int, int]]:
    """Return (preprocessed image, original (height, width))."""
    rgb = load_image_rgb(path)
    return preprocess_array(rgb), (int(rgb.shape[0]), int(rgb.shape[1]))


def preprocess_image(path: str | Path) -> np.ndarray:
    """Full pipeline for one file. Output shape: (128, 128, 3), values in [0, 1]."""
    return load_and_preprocess(path)[0]


# --------------------------------------------------------------------------- #
# Duplicate helpers (used for leakage control and dataset inspection)
# --------------------------------------------------------------------------- #
def file_md5(path: str | Path) -> str:
    """MD5 of the raw file bytes (detects byte-identical duplicates)."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(image01: np.ndarray) -> np.ndarray:
    """64-bit difference hash of a preprocessed image (bool array of length 64)."""
    gray = cv2.cvtColor((image01 * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).flatten()


def group_near_duplicates(hashes: np.ndarray, max_distance: int) -> list[list[int]]:
    """Group indices whose dHash Hamming distance is <= max_distance.

    Returns a list of groups (every index appears in exactly one group).
    Singletons are returned as groups of size 1.
    """
    n = len(hashes)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if n > 1:
        h = hashes.astype(np.float32)
        ones = h.sum(axis=1)
        chunk = 1024
        for start in range(0, n, chunk):
            block = h[start:start + chunk]
            dist = ones[start:start + chunk, None] + ones[None, :] - 2.0 * (block @ h.T)
            rows, cols = np.where(dist <= max_distance + 0.5)
            for r, c in zip(rows, cols):
                i, j = start + int(r), int(c)
                if i != j:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[ri] = rj

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())
