"""Inspect the dataset before training.

Usage:
    python src/check_dataset.py
    python src/check_dataset.py --preview 12         # save a grid of preprocessed images
    python src/check_dataset.py --augmentation       # save augmentation examples
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src import config
from src.preprocessing import (
    dhash,
    discover_images,
    file_md5,
    group_near_duplicates,
    load_image_rgb,
    preprocess_array,
)


def analyse_class(folder: Path) -> dict:
    """Read every image once; collect sizes, hashes and corrupted files."""
    paths = discover_images(folder)
    info = {"folder": folder, "found": len(paths), "ok_paths": [], "corrupted": [],
            "sizes": [], "md5": [], "dhash": [], "preview_cache": {}}
    for i, path in enumerate(paths, start=1):
        try:
            rgb = load_image_rgb(path)
            small = preprocess_array(rgb)
            info["sizes"].append((int(rgb.shape[0]), int(rgb.shape[1])))
            info["md5"].append(file_md5(path))
            info["dhash"].append(dhash(small))
            info["ok_paths"].append(path)
        except Exception as exc:  # noqa: BLE001
            info["corrupted"].append((path, str(exc)))
        if i % 200 == 0:
            print(f"  ... checked {i}/{len(paths)} in {folder.name}")
    return info


def print_class_report(title: str, info: dict) -> None:
    print(f"\n{title}")
    print(f"Folder : {info['folder']}")
    print(f"Images : {info['found']}  (readable: {len(info['ok_paths'])}, "
          f"corrupted: {len(info['corrupted'])})")
    for path, reason in info["corrupted"][:10]:
        print(f"   CORRUPTED: {path.name} -> {reason}")
    if len(info["corrupted"]) > 10:
        print(f"   ... and {len(info['corrupted']) - 10} more corrupted files")

    if info["sizes"]:
        heights = np.array([s[0] for s in info["sizes"]])
        widths = np.array([s[1] for s in info["sizes"]])
        print("Original image size BEFORE preprocessing (pixels):")
        print(f"   height: min={heights.min()}  median={int(np.median(heights))}  max={heights.max()}")
        print(f"   width : min={widths.min()}  median={int(np.median(widths))}  max={widths.max()}")
        print(f"   portrait={(heights > widths).sum()}  landscape={(widths > heights).sum()}  "
              f"square={(heights == widths).sum()}")
        small = int(((heights < 64) | (widths < 64)).sum())
        if small:
            print(f"   WARNING: {small} image(s) are smaller than 64 px on a side (very low detail).")

    md5s = info["md5"]
    exact = len(md5s) - len(set(md5s))
    print(f"Exact duplicate files : {exact}")
    if info["dhash"]:
        groups = group_near_duplicates(np.stack(info["dhash"]), config.NEAR_DUPLICATE_MAX_HAMMING)
        multi = [g for g in groups if len(g) > 1]
        extra = sum(len(g) - 1 for g in multi)
        print(f"Near-duplicate groups : {len(multi)} ({extra} redundant images)")
        for g in multi[:5]:
            print("   - " + ", ".join(info["ok_paths"][i].name for i in g[:4])
                  + (" ..." if len(g) > 4 else ""))
        if multi:
            print("   Near duplicates inflate accuracy if some land in train and others in test.\n"
                  "   train.py keeps each group inside ONE split, but consider removing them.")


def save_preview(infos: list[dict], count: int, show: bool) -> None:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(config.RANDOM_SEED)
    pool = [(p, info["folder"].name) for info in infos for p in info["ok_paths"]]
    if not pool:
        print("Nothing to preview.")
        return
    chosen = [pool[int(i)] for i in rng.permutation(len(pool))[:count]]
    cols = min(6, len(chosen))
    rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (path, cls) in zip(axes.ravel(), chosen):
        ax.imshow(preprocess_array(load_image_rgb(path)))
        ax.set_title(f"{cls}\n128x128x3", fontsize=8)
    fig.tight_layout()
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.OUTPUTS_DIR / "preview_images.png"
    fig.savefig(out, dpi=120)
    print(f"Saved preview: {out}")
    if show:
        plt.show()
    plt.close(fig)


def save_augmentation_examples(infos: list[dict], show: bool) -> None:
    import matplotlib.pyplot as plt
    import tensorflow as tf

    from src.model import build_augmentation

    pool = [p for info in infos for p in info["ok_paths"]]
    if not pool:
        print("Nothing to augment.")
        return
    config.set_global_seed()
    image = preprocess_array(load_image_rgb(pool[0]))
    aug = build_augmentation()
    fig, axes = plt.subplots(2, 4, figsize=(9, 5))
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("original")
    for ax in axes.ravel()[1:]:
        out = aug(tf.constant(image[None, ...]), training=True)[0].numpy()
        ax.imshow(np.clip(out, 0.0, 1.0))
        ax.set_title("augmented")
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.OUTPUTS_DIR / "augmentation_examples.png"
    fig.savefig(out_path, dpi=120)
    print(f"Saved augmentation examples: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect the saree dataset.")
    parser.add_argument("--preview", type=int, default=0, metavar="N",
                        help="save a grid of N preprocessed 128x128 images")
    parser.add_argument("--augmentation", action="store_true",
                        help="save an image of augmentation examples")
    parser.add_argument("--show", action="store_true", help="also open the plot windows")
    args = parser.parse_args()

    config.ensure_directories()
    print("=" * 60)
    print("DATASET CHECK")
    print("=" * 60)

    pos = analyse_class(config.POSITIVE_DIR)
    neg = analyse_class(config.NEGATIVE_DIR)

    print_class_report("MARATHI DATASET", pos)
    print_class_report("NON-MARATHI DATASET", neg)

    n_pos, n_neg = len(pos["ok_paths"]), len(neg["ok_paths"])
    print("\n" + "-" * 60)
    print(f"Marathi images     : {n_pos}")
    print(f"Non-Marathi images : {n_neg}")
    print(f"Total (readable)   : {n_pos + n_neg}")
    print(f"\nImage preprocessing:\n{config.IMAGE_SIZE[0]} × {config.IMAGE_SIZE[1]} × {config.CHANNELS}")

    cross = set(pos["md5"]) & set(neg["md5"])
    if cross:
        print(f"\nWARNING: {len(cross)} identical file(s) appear in BOTH classes (label conflict).")

    problems = 0
    if n_pos == 0:
        problems += 1
        print(f"\nERROR: no readable images in {config.POSITIVE_DIR}")
        print("Put your Marathi saree images (.jpg/.jpeg/.png/.webp) into that folder.")
    elif n_pos < config.MIN_IMAGES_PER_CLASS:
        problems += 1
        print(f"\nERROR: need at least {config.MIN_IMAGES_PER_CLASS} Marathi images.")

    if n_pos and n_neg == 0:
        print("\nWARNING:")
        print("Only one class is available.")
        print("A normal binary CNN classifier cannot properly learn Marathi vs Non-Marathi")
        print("from positive examples alone.")
        print(f"Recommendation: add non-Marathi saree images to\n  {config.NEGATIVE_DIR}")
        print("`python src/train.py` will use the weaker POSITIVE-ONLY fallback until then.")
    elif n_pos and n_neg:
        if n_neg < config.MIN_IMAGES_PER_CLASS:
            problems += 1
            print(f"\nERROR: need at least {config.MIN_IMAGES_PER_CLASS} non-Marathi images.")
        ratio = max(n_pos, n_neg) / max(1, min(n_pos, n_neg))
        print(f"\nClass imbalance ratio (majority/minority): {ratio:.2f}")
        if ratio > 1.5:
            print("WARNING: classes are imbalanced. train.py applies class weights, but a "
                  "balanced dataset is better.")

    if problems == 0 and n_pos:
        mode = "SUPERVISED BINARY CLASSIFICATION" if n_neg else "POSITIVE-ONLY FALLBACK"
        print(f"\nTraining mode that `python src/train.py` will choose: {mode}")

    if args.preview > 0:
        save_preview([pos, neg], args.preview, args.show)
    if args.augmentation:
        save_augmentation_examples([pos, neg], args.show)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
