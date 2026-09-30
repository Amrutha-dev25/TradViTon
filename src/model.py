"""CNN architecture, feature extractor, and positive-only (one-class) helpers."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src import config

import tensorflow as tf  # noqa: E402  (after config sets log level)
from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402

FEATURE_LAYER_NAME = "feature_vector"


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #
def build_augmentation() -> keras.Sequential:
    """Moderate augmentation that keeps the saree drape recognisable.

    No vertical flips, no large rotations, no strong colour shifts / hue changes.
    """
    aug = []
    if config.USE_HORIZONTAL_FLIP:
        aug.append(layers.RandomFlip("horizontal"))
    aug += [
        layers.RandomRotation(config.AUG_ROTATION, fill_mode="reflect"),
        layers.RandomZoom(height_factor=config.AUG_ZOOM, fill_mode="reflect"),
        layers.RandomTranslation(config.AUG_TRANSLATION, config.AUG_TRANSLATION,
                                 fill_mode="reflect"),
        layers.RandomBrightness(config.AUG_BRIGHTNESS, value_range=(0.0, 1.0)),
    ]
    return keras.Sequential(aug, name="augmentation")


# --------------------------------------------------------------------------- #
# Architecture
# --------------------------------------------------------------------------- #
def _build_encoder(inputs):
    """Conv/Pool x3 -> Flatten -> Dense(256) -> Dense(128) = FEATURE VECTOR."""
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv2d_1")(inputs)
    x = layers.MaxPooling2D((2, 2), name="max_pooling2d_1")(x)          # 128 -> 64
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2d_2")(x)
    x = layers.MaxPooling2D((2, 2), name="max_pooling2d_2")(x)          # 64 -> 32
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv2d_3")(x)
    x = layers.MaxPooling2D((2, 2), name="max_pooling2d_3")(x)          # 32 -> 16
    x = layers.Flatten(name="flatten")(x)                                # 16*16*128 = 32768
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    features = layers.Dense(config.FEATURE_VECTOR_SIZE, activation="relu",
                            name=FEATURE_LAYER_NAME)(x)                  # <- 128-d feature vector
    return features


def build_classifier() -> keras.Model:
    """Supervised binary classifier: ... -> feature_vector -> Dropout -> Dense(1, sigmoid)."""
    inputs = keras.Input(shape=config.INPUT_SHAPE, name="input_image")
    features = _build_encoder(inputs)
    x = layers.Dropout(config.DROPOUT_RATE, name="dropout")(features)
    outputs = layers.Dense(1, activation="sigmoid", name="classifier_output")(x)
    return keras.Model(inputs, outputs, name="marathi_saree_cnn")


def build_autoencoder() -> keras.Model:
    """Positive-only representation learner.

    Same encoder (so the same 128-d `feature_vector` layer) followed by a decoder
    that tries to reconstruct the input. This needs NO negative examples.
    """
    if config.IMAGE_SIZE[0] % 8 or config.IMAGE_SIZE[1] % 8:
        raise ValueError("IMAGE_SIZE must be divisible by 8 for the decoder.")
    h, w = config.IMAGE_SIZE[0] // 8, config.IMAGE_SIZE[1] // 8

    inputs = keras.Input(shape=config.INPUT_SHAPE, name="input_image")
    features = _build_encoder(inputs)
    x = layers.Dense(256, activation="relu", name="decoder_dense_256")(features)
    x = layers.Dense(h * w * 64, activation="relu", name="decoder_dense_expand")(x)
    # Reshape here is a network layer (unflattening features), NOT image preprocessing.
    x = layers.Reshape((h, w, 64), name="decoder_unflatten")(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, padding="same",
                               activation="relu", name="decoder_deconv_1")(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, padding="same",
                               activation="relu", name="decoder_deconv_2")(x)
    x = layers.Conv2DTranspose(16, (3, 3), strides=2, padding="same",
                               activation="relu", name="decoder_deconv_3")(x)
    outputs = layers.Conv2D(3, (3, 3), padding="same", activation="sigmoid",
                            name="reconstruction")(x)
    return keras.Model(inputs, outputs, name="marathi_saree_autoencoder")


def build_feature_extractor(model: keras.Model) -> keras.Model:
    """Model that maps an image (128,128,3) to the 128-d `feature_vector` layer."""
    return keras.Model(
        inputs=model.input,
        outputs=model.get_layer(FEATURE_LAYER_NAME).output,
        name="feature_extractor",
    )


def print_model_summary(model: keras.Model) -> None:
    """Print the architecture (Conv2D, MaxPooling2D, Flatten, Dense, feature_vector, ...)."""
    print("\n" + "=" * 70)
    print(f"MODEL ARCHITECTURE: {model.name}")
    print("=" * 70)
    model.summary()
    print(f"\nThe layer named '{FEATURE_LAYER_NAME}' outputs the "
          f"{config.FEATURE_VECTOR_SIZE}-dimensional feature vector.\n")


# --------------------------------------------------------------------------- #
# Saving / loading artifacts
# --------------------------------------------------------------------------- #
def save_training_mode(mode: str, extra: dict | None = None) -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"mode": mode, **(extra or {})}
    with open(config.TRAINING_MODE_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_training_mode() -> dict:
    if not config.TRAINING_MODE_PATH.exists():
        raise FileNotFoundError("No trained model found. Run `python src/train.py` first.")
    with open(config.TRAINING_MODE_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_classifier() -> keras.Model:
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing {config.MODEL_PATH}. Run `python src/train.py` first.")
    return keras.models.load_model(config.MODEL_PATH, compile=False)


def load_feature_extractor() -> keras.Model:
    if not config.FEATURE_EXTRACTOR_PATH.exists():
        raise FileNotFoundError(
            f"Missing {config.FEATURE_EXTRACTOR_PATH}. Run `python src/train.py` first.")
    return keras.models.load_model(config.FEATURE_EXTRACTOR_PATH, compile=False)


# --------------------------------------------------------------------------- #
# Positive-only (one-class) helpers
# --------------------------------------------------------------------------- #
def cosine_distance(features: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """1 - cosine similarity between each feature vector and the centroid."""
    f = np.atleast_2d(np.asarray(features, dtype=np.float64))
    c = np.asarray(centroid, dtype=np.float64).ravel()
    denom = np.linalg.norm(f, axis=1) * np.linalg.norm(c) + 1e-12
    return 1.0 - (f @ c) / denom


def fit_one_class(train_features: np.ndarray, val_features: np.ndarray) -> dict:
    """Centroid from TRAIN features; threshold from HELD-OUT validation distances."""
    centroid = train_features.mean(axis=0)
    val_dist = cosine_distance(val_features, centroid)
    train_dist = cosine_distance(train_features, centroid)
    threshold = float(np.percentile(val_dist, config.ONE_CLASS_PERCENTILE))
    return {
        "centroid": centroid.astype(np.float32),
        "threshold": threshold,
        "percentile": float(config.ONE_CLASS_PERCENTILE),
        "val_distance_mean": float(val_dist.mean()),
        "val_distance_max": float(val_dist.max()),
        "train_distance_mean": float(train_dist.mean()),
    }


def save_one_class_stats(stats: dict) -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(config.ONE_CLASS_STATS_PATH, **{k: np.asarray(v) for k, v in stats.items()})


def load_one_class_stats() -> dict:
    if not config.ONE_CLASS_STATS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {config.ONE_CLASS_STATS_PATH}. Run `python src/train.py` first.")
    with np.load(config.ONE_CLASS_STATS_PATH) as data:
        stats = {k: data[k] for k in data.files}
    stats["threshold"] = float(stats["threshold"])
    return stats


if __name__ == "__main__":
    print_model_summary(build_classifier())
