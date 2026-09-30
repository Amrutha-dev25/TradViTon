"""Classify an external image (it never needs to be inside dataset/).

Usage:  python src/predict.py --image test_images/example.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src import config
from src.model import (
    cosine_distance,
    load_classifier,
    load_feature_extractor,
    load_one_class_stats,
    load_training_mode,
)
from src.preprocessing import preprocess_image

LINE = "=" * 50


def predict_binary(model, image: np.ndarray, confidence_threshold: float) -> dict:
    p_marathi = float(model.predict(image[None, ...], verbose=0)[0, 0])
    p_non = 1.0 - p_marathi
    confidence = max(p_marathi, p_non)
    if confidence < confidence_threshold:
        label = "UNCERTAIN"
    elif p_marathi >= config.CLASSIFICATION_THRESHOLD:
        label = "MARATHI / MAHARASHTRA STYLE"
    else:
        label = "NON-MARATHI STYLE"
    return {"mode": "binary", "label": label, "confidence": confidence,
            "p_marathi": p_marathi, "p_non": p_non}


def predict_positive_only(extractor, stats: dict, image: np.ndarray) -> dict:
    feature = extractor.predict(image[None, ...], verbose=0)[0]
    distance = float(cosine_distance(feature, stats["centroid"])[0])
    threshold = stats["threshold"]
    margin = config.ONE_CLASS_UNCERTAIN_MARGIN
    if distance <= threshold * (1.0 - margin):
        label = "LIKELY MARATHI"
    elif distance >= threshold * (1.0 + margin):
        label = "LIKELY NON-MARATHI / OUTSIDE TRAINING DISTRIBUTION"
    else:
        label = "UNCERTAIN (close to the decision threshold)"
    return {"mode": "positive_only", "label": label, "distance": distance,
            "threshold": threshold}


def print_result(image_name: str, result: dict) -> None:
    print("\n" + LINE)
    print("MARATHI SAREE CLASSIFICATION")
    print(LINE)
    print(f"\nImage: {image_name}\n")
    print(f"Prediction: {result['label']}\n")
    if result["mode"] == "binary":
        print(f"Confidence: {result['confidence'] * 100:.1f}%\n")
        print("Probability:")
        print(f"Marathi: {result['p_marathi']:.3f}")
        print(f"Non-Marathi: {result['p_non']:.3f}\n")
        if result["label"] == "UNCERTAIN":
            print("The model is not confident enough to give a strong answer. The image may")
            print("differ a lot from the training data (pose, background, another style).\n")
    else:
        print("Mode: POSITIVE-ONLY (weaker than a supervised binary classifier)")
        print(f"Cosine distance to Marathi centroid: {result['distance']:.4f}")
        print(f"Decision threshold                 : {result['threshold']:.4f}")
        print("This is a distance, NOT a probability. Add dataset/non_maharashtra/")
        print("and retrain for a proper Marathi vs Non-Marathi classifier.\n")
    print(LINE + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict Marathi vs Non-Marathi saree style.")
    parser.add_argument("--image", required=True, type=Path, help="path to an image file")
    parser.add_argument("--confidence-threshold", type=float,
                        default=config.CONFIDENCE_THRESHOLD,
                        help="below this confidence the answer is UNCERTAIN (binary mode)")
    args = parser.parse_args()

    if not args.image.is_file():
        print(f"ERROR: image not found: {args.image}")
        return 1
    try:
        image = preprocess_image(args.image)          # same pipeline as training
        mode = load_training_mode()["mode"]
        if mode == "binary":
            result = predict_binary(load_classifier(), image, args.confidence_threshold)
        else:
            result = predict_positive_only(load_feature_extractor(),
                                           load_one_class_stats(), image)
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print_result(args.image.name, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
