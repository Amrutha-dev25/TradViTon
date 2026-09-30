"""Extract the 128-dimensional CNN feature vector for an external image.

Usage:
    python src/feature_extractor.py --image test_images/example.jpg
    python src/feature_extractor.py --image test_images/example.jpg --save
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src import config
from src.model import load_feature_extractor
from src.preprocessing import preprocess_image


def extract_feature_vector(extractor, image: np.ndarray) -> np.ndarray:
    """image: (128,128,3) float32 in [0,1]  ->  feature vector of shape (128,)."""
    batch = np.expand_dims(image, axis=0)              # (1,128,128,3)
    return extractor.predict(batch, verbose=0)[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract the 128-d feature vector of an image.")
    parser.add_argument("--image", required=True, type=Path, help="path to an image file")
    parser.add_argument("--save", action="store_true",
                        help="save the vector to outputs/<image_name>_feature.npy")
    parser.add_argument("--show-all", action="store_true", help="print all 128 values")
    args = parser.parse_args()

    if not args.image.is_file():
        print(f"ERROR: image not found: {args.image}")
        return 1
    try:
        image = preprocess_image(args.image)
        extractor = load_feature_extractor()
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1

    vector = extract_feature_vector(extractor, image)
    print(f"Input after preprocessing : {image.shape}")
    print(f"Feature vector shape: {vector.shape}")
    print(f"Preview (first 8 values): {np.round(vector[:8], 4)}")
    print(f"Min={vector.min():.4f}  Max={vector.max():.4f}  Mean={vector.mean():.4f}")
    if args.show_all:
        np.set_printoptions(precision=4, suppress=True, linewidth=100)
        print(vector)
    if args.save:
        config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = config.OUTPUTS_DIR / f"{args.image.stem}_feature.npy"
        np.save(out_path, vector)
        print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
