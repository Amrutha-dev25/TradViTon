"""Train the model. Automatically picks the workflow from the dataset folders.

    dataset/maharashtra + dataset/non_maharashtra  -> supervised binary CNN
    dataset/maharashtra only                       -> positive-only fallback
                                                      (autoencoder features + centroid distance)

Usage:  python src/train.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src import config
from src.dataset import (
    MODE_BINARY,
    DatasetError,
    compute_class_weights,
    create_splits,
    make_dataset,
    print_split_summary,
)

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402

from src.model import (  # noqa: E402
    build_augmentation,
    build_autoencoder,
    build_classifier,
    build_feature_extractor,
    fit_one_class,
    print_model_summary,
    save_one_class_stats,
    save_training_mode,
)


def make_callbacks() -> list:
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.EARLY_STOPPING_PATIENCE,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(filepath=str(config.MODEL_PATH), monitor="val_loss",
                                        save_best_only=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=config.LR_REDUCE_FACTOR,
                                          patience=config.LR_REDUCE_PATIENCE,
                                          min_lr=config.MIN_LEARNING_RATE, verbose=1),
    ]


def plot_history(history: dict, binary: bool, path: Path) -> None:
    """Save training/validation accuracy (or MAE) and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    metric, title = ("accuracy", "Accuracy") if binary else ("mae", "Reconstruction MAE")
    axes[0].plot(history[metric], label=f"train {metric}")
    axes[0].plot(history[f"val_{metric}"], label=f"val {metric}")
    axes[0].set_title(title)
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(history["loss"], label="train loss")
    axes[1].plot(history["val_loss"], label="val loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {path}")


def train_binary(data) -> None:
    print("\n>>> MODE: SUPERVISED BINARY CLASSIFICATION (maharashtra=1, non_maharashtra=0)")
    model = build_classifier()
    print_model_summary(model)
    model.compile(optimizer=keras.optimizers.Adam(config.LEARNING_RATE),
                  loss="binary_crossentropy", metrics=["accuracy"])

    augmentation = build_augmentation()
    train_ds = make_dataset(data.X_train, data.y_train, training=True, augmentation=augmentation)
    val_ds = make_dataset(data.X_val, data.y_val, training=False)
    class_weight = compute_class_weights(data.y_train)
    print(f"Class weights (handles imbalance): {class_weight}")

    history = model.fit(train_ds, validation_data=val_ds, epochs=config.EPOCHS,
                        class_weight=class_weight, callbacks=make_callbacks(), verbose=1)

    model.save(config.MODEL_PATH)  # final (best-weights) model
    extractor = build_feature_extractor(model)
    extractor.save(config.FEATURE_EXTRACTOR_PATH)
    save_training_mode(MODE_BINARY)
    plot_history(history.history, True, config.OUTPUTS_DIR / "training_history.png")
    with open(config.OUTPUTS_DIR / "history.json", "w", encoding="utf-8") as handle:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, handle)

    print(f"\nSaved model            : {config.MODEL_PATH}")
    print(f"Saved feature extractor: {config.FEATURE_EXTRACTOR_PATH}")
    print("Next: python src/evaluate.py   (this reports the REAL test-set results)")


def train_positive_only(data) -> None:
    print("\n" + "!" * 70)
    print("WARNING: only the 'maharashtra' class is available.")
    print("A normal binary classifier cannot learn Marathi vs Non-Marathi from")
    print("positive examples alone, so NO sigmoid classifier is trained.")
    print("Fallback: an autoencoder learns a 128-d feature vector from Marathi")
    print("images; new images are judged by distance to the Marathi centroid.")
    print("This is WEAKER than supervised training. Add dataset/non_maharashtra/.")
    print("!" * 70)

    model = build_autoencoder()
    print_model_summary(model)
    model.compile(optimizer=keras.optimizers.Adam(config.LEARNING_RATE), loss="mse",
                  metrics=["mae"])

    augmentation = build_augmentation()
    train_ds = make_dataset(data.X_train, None, training=True, augmentation=augmentation)
    val_ds = make_dataset(data.X_val, None, training=False)
    history = model.fit(train_ds, validation_data=val_ds, epochs=config.EPOCHS,
                        callbacks=make_callbacks(), verbose=1)

    model.save(config.MODEL_PATH)
    extractor = build_feature_extractor(model)
    extractor.save(config.FEATURE_EXTRACTOR_PATH)

    train_feats = extractor.predict(data.X_train, batch_size=config.BATCH_SIZE, verbose=0)
    val_feats = extractor.predict(data.X_val, batch_size=config.BATCH_SIZE, verbose=0)
    stats = fit_one_class(train_feats, val_feats)
    save_one_class_stats(stats)
    save_training_mode("positive_only")
    plot_history(history.history, False, config.OUTPUTS_DIR / "training_history.png")

    print(f"\nSaved autoencoder      : {config.MODEL_PATH}")
    print(f"Saved feature extractor: {config.FEATURE_EXTRACTOR_PATH}")
    print(f"Saved one-class stats  : {config.ONE_CLASS_STATS_PATH}")
    print(f"Cosine-distance threshold = {stats['threshold']:.4f} "
          f"({config.ONE_CLASS_PERCENTILE}th percentile of held-out Marathi images)")
    print("Next: python src/evaluate.py")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the Marathi saree model.")
    parser.add_argument("--epochs", type=int, default=None, help="override config.EPOCHS")
    args = parser.parse_args()
    if args.epochs is not None:
        config.EPOCHS = args.epochs

    config.ensure_directories()
    config.set_global_seed()
    print(f"TensorFlow {tf.__version__} | GPUs: {len(tf.config.list_physical_devices('GPU'))}")

    try:
        data = create_splits(save_manifest=True)
    except DatasetError as exc:
        print(f"\nDATASET ERROR: {exc}")
        print("Run `python src/check_dataset.py` for details.")
        return 1
    print_split_summary(data)

    if data.mode == MODE_BINARY:
        train_binary(data)
    else:
        train_positive_only(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
