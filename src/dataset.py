"""Dataset discovery, folder-based labelling, leakage-aware splitting, tf.data.

Labels come from folder names:
    dataset/maharashtra/      -> 1
    dataset/non_maharashtra/  -> 0   (optional)
No CSV is needed.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src import config
from src.preprocessing import (
    dhash,
    discover_images,
    file_md5,
    group_near_duplicates,
    load_and_preprocess,
)

MODE_BINARY = "binary"
MODE_POSITIVE_ONLY = "positive_only"


class DatasetError(Exception):
    """Raised when the dataset cannot be used (with a user-friendly message)."""


@dataclass
class SplitData:
    mode: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    paths_train: list[str]
    paths_val: list[str]
    paths_test: list[str]
    info: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Discovery / mode detection
# --------------------------------------------------------------------------- #
def discover_dataset() -> dict[str, list[Path]]:
    return {
        "positive": discover_images(config.POSITIVE_DIR),
        "negative": discover_images(config.NEGATIVE_DIR),
    }


def detect_mode(found: dict[str, list[Path]]) -> str:
    if not found["positive"]:
        raise DatasetError(
            f"No images found in {config.POSITIVE_DIR}.\n"
            f"Put your Marathi saree images (.jpg/.jpeg/.png/.webp) there."
        )
    if not found["negative"]:
        return MODE_POSITIVE_ONLY
    return MODE_BINARY


def _rel(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(config.PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _abs(rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    return p if p.is_absolute() else config.PROJECT_ROOT / p


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _load_records(paths: list[Path], label: int, failures: list[tuple[str, str]]) -> list[dict]:
    records = []
    total = len(paths)
    for i, path in enumerate(paths, start=1):
        try:
            image, _ = load_and_preprocess(path)
            records.append({
                "path": path, "image": image, "label": label,
                "md5": file_md5(path), "dhash": dhash(image),
            })
        except Exception as exc:  # noqa: BLE001 - report, never crash training
            failures.append((str(path), str(exc)))
        if i % 100 == 0 or i == total:
            print(f"  loaded {i}/{total} images (label={label})")
    return records


def load_images(paths: list[Path]) -> tuple[np.ndarray, list[Path], list[tuple[str, str]]]:
    """Load and preprocess many files. Returns (X, kept_paths, failures)."""
    images, kept, failures = [], [], []
    for path in paths:
        try:
            images.append(load_and_preprocess(path)[0])
            kept.append(path)
        except Exception as exc:  # noqa: BLE001
            failures.append((str(path), str(exc)))
    if images:
        X = np.stack(images).astype(np.float32)
    else:
        X = np.zeros((0, *config.INPUT_SHAPE), dtype=np.float32)
    return X, kept, failures


# --------------------------------------------------------------------------- #
# Splitting (stratified per class, near-duplicates stay together)
# --------------------------------------------------------------------------- #
def _split_class(records: list[dict], rng: np.random.Generator, name: str):
    if len(records) < config.MIN_IMAGES_PER_CLASS:
        raise DatasetError(
            f"Class '{name}' has only {len(records)} usable images; "
            f"at least {config.MIN_IMAGES_PER_CLASS} are required."
        )
    hashes = np.stack([r["dhash"] for r in records])
    groups = group_near_duplicates(hashes, config.NEAR_DUPLICATE_MAX_HAMMING)
    n = len(records)
    n_test = max(1, round(n * config.TEST_FRACTION))
    n_val = max(1, round(n * config.VAL_FRACTION))

    train, val, test = [], [], []
    for gi in rng.permutation(len(groups)):
        members = [records[i] for i in groups[int(gi)]]
        if len(test) < n_test:
            test.extend(members)
        elif len(val) < n_val:
            val.extend(members)
        else:
            train.extend(members)
    if not train or not val or not test:
        raise DatasetError(
            f"Could not build non-empty train/val/test splits for class '{name}'. "
            f"Too many near-duplicate images? Add more varied images."
        )
    return train, val, test


def _to_arrays(records: list[dict], rng: np.random.Generator):
    order = rng.permutation(len(records))
    records = [records[int(i)] for i in order]
    X = np.stack([r["image"] for r in records]).astype(np.float32)
    y = np.array([r["label"] for r in records], dtype=np.float32)
    paths = [_rel(r["path"]) for r in records]
    return X, y, paths


def create_splits(save_manifest: bool = True) -> SplitData:
    """Discover images, drop exact duplicates, split, and save a manifest.

    The manifest (models/split_manifest.json) lets evaluate.py reuse the very
    same test set later, even if you add images to the dataset in between.
    """
    found = discover_dataset()
    mode = detect_mode(found)
    rng = np.random.default_rng(config.RANDOM_SEED)
    failures: list[tuple[str, str]] = []

    print(f"Loading images ({'2 classes' if mode == MODE_BINARY else '1 class'}) ...")
    per_class = {1: _load_records(found["positive"], 1, failures)}
    if mode == MODE_BINARY:
        per_class[0] = _load_records(found["negative"], 0, failures)

    # Exact duplicates (byte-identical files) are removed -> they would leak.
    seen: set[str] = set()
    n_exact = 0
    for label in per_class:
        unique = []
        for rec in per_class[label]:
            if rec["md5"] in seen:
                n_exact += 1
                continue
            seen.add(rec["md5"])
            unique.append(rec)
        per_class[label] = unique

    parts = {"train": [], "val": [], "test": []}
    for label, records in per_class.items():
        name = config.POSITIVE_CLASS_NAME if label == 1 else config.NEGATIVE_CLASS_NAME
        tr, va, te = _split_class(records, rng, name)
        parts["train"] += tr
        parts["val"] += va
        parts["test"] += te

    X_tr, y_tr, p_tr = _to_arrays(parts["train"], rng)
    X_va, y_va, p_va = _to_arrays(parts["val"], rng)
    X_te, y_te, p_te = _to_arrays(parts["test"], rng)

    info = {
        "corrupted_files": failures,
        "exact_duplicates_removed": n_exact,
        "images_per_class": {int(k): len(v) for k, v in per_class.items()},
    }
    data = SplitData(mode, X_tr, y_tr, X_va, y_va, X_te, y_te, p_tr, p_va, p_te, info)

    if save_manifest:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        manifest = {
            "mode": mode,
            "seed": config.RANDOM_SEED,
            "splits": {
                "train": [[p, int(l)] for p, l in zip(p_tr, y_tr)],
                "val": [[p, int(l)] for p, l in zip(p_va, y_va)],
                "test": [[p, int(l)] for p, l in zip(p_te, y_te)],
            },
        }
        with open(config.SPLIT_MANIFEST_PATH, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
    return data


def load_splits_from_manifest() -> SplitData:
    """Rebuild the exact train/val/test split saved by train.py."""
    if not config.SPLIT_MANIFEST_PATH.exists():
        raise DatasetError("No split manifest found. Run `python src/train.py` first.")
    with open(config.SPLIT_MANIFEST_PATH, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    failures: list[tuple[str, str]] = []
    arrays = {}
    for split in ("train", "val", "test"):
        images, labels, paths = [], [], []
        for rel, label in manifest["splits"][split]:
            try:
                images.append(load_and_preprocess(_abs(rel))[0])
                labels.append(label)
                paths.append(rel)
            except Exception as exc:  # noqa: BLE001
                failures.append((rel, str(exc)))
        X = (np.stack(images).astype(np.float32) if images
             else np.zeros((0, *config.INPUT_SHAPE), dtype=np.float32))
        arrays[split] = (X, np.array(labels, dtype=np.float32), paths)

    return SplitData(
        manifest["mode"],
        *arrays["train"][:2], *arrays["val"][:2], *arrays["test"][:2],
        arrays["train"][2], arrays["val"][2], arrays["test"][2],
        info={"corrupted_files": failures},
    )


def print_split_summary(data: SplitData) -> None:
    print("\nDATA SPLIT (stratified per class, near-duplicates kept together)")
    for name, y in (("train", data.y_train), ("val", data.y_val), ("test", data.y_test)):
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        print(f"  {name:<5}: {len(y):>5} images  (Marathi={pos}, Non-Marathi={neg})")
    print(f"  Exact duplicates removed: {data.info.get('exact_duplicates_removed', 0)}")
    bad = data.info.get("corrupted_files", [])
    print(f"  Unreadable/corrupted files skipped: {len(bad)}")
    for path, reason in bad[:10]:
        print(f"    - {path}: {reason}")
    if len(bad) > 10:
        print(f"    ... and {len(bad) - 10} more")


def compute_class_weights(y_train: np.ndarray) -> dict[int, float]:
    """Balanced class weights for the supervised binary mode."""
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.array([0, 1])
    weights = compute_class_weight("balanced", classes=classes, y=y_train.astype(int))
    return {int(c): float(w) for c, w in zip(classes, weights)}


# --------------------------------------------------------------------------- #
# tf.data pipelines
# --------------------------------------------------------------------------- #
def make_dataset(X: np.ndarray, y: np.ndarray | None = None,
                 training: bool = False, augmentation=None):
    """Build a batched tf.data pipeline.

    y is None  -> autoencoder mode: targets are the (augmented) inputs.
    Augmentation is applied only when training=True.
    """
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices(X if y is None else (X, y))
    if training:
        ds = ds.shuffle(len(X), seed=config.RANDOM_SEED, reshuffle_each_iteration=True)
    ds = ds.batch(config.BATCH_SIZE)

    use_aug = training and augmentation is not None

    def _aug(batch):
        return tf.clip_by_value(augmentation(batch, training=True), 0.0, 1.0)

    if y is None:
        if use_aug:
            def _fn(batch):
                a = _aug(batch)
                return a, a
        else:
            def _fn(batch):
                return batch, batch
    else:
        if use_aug:
            def _fn(batch, target):
                return _aug(batch), target
        else:
            def _fn(batch, target):
                return batch, target

    return ds.map(_fn, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
