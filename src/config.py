"""Central configuration for the Marathi saree classifier.

Every tunable value lives here so that no other file hardcodes it.
This module deliberately does NOT import TensorFlow at import time, so light
scripts (e.g. check_dataset.py) start quickly.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

# Quieter TensorFlow logs (must be set before TensorFlow is imported anywhere).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# --------------------------------------------------------------------------- #
# Paths (pathlib only -> works on Windows, macOS and Linux)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
POSITIVE_CLASS_NAME = "maharashtra"          # class 1 (folder name = label)
NEGATIVE_CLASS_NAME = "non_maharashtra"      # class 0 (folder name = label)
POSITIVE_DIR = DATASET_DIR / POSITIVE_CLASS_NAME
NEGATIVE_DIR = DATASET_DIR / NEGATIVE_CLASS_NAME

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TEST_IMAGES_DIR = PROJECT_ROOT / "test_images"

MODEL_PATH = MODELS_DIR / "marathi_cnn.keras"
FEATURE_EXTRACTOR_PATH = MODELS_DIR / "feature_extractor.keras"
ONE_CLASS_STATS_PATH = MODELS_DIR / "one_class_stats.npz"
TRAINING_MODE_PATH = MODELS_DIR / "training_mode.json"
SPLIT_MANIFEST_PATH = MODELS_DIR / "split_manifest.json"

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

# --------------------------------------------------------------------------- #
# Image preprocessing
# --------------------------------------------------------------------------- #
IMAGE_SIZE = (128, 128)                      # (height, width)
CHANNELS = 3
INPUT_SHAPE = (IMAGE_SIZE[0], IMAGE_SIZE[1], CHANNELS)

# --------------------------------------------------------------------------- #
# Model / training
# --------------------------------------------------------------------------- #
FEATURE_VECTOR_SIZE = 128
DROPOUT_RATE = 0.5
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

EARLY_STOPPING_PATIENCE = 8
LR_REDUCE_FACTOR = 0.5
LR_REDUCE_PATIENCE = 3
MIN_LEARNING_RATE = 1e-6

# Data split (fractions of every class)
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
MIN_IMAGES_PER_CLASS = 20

# Near-duplicate detection (difference hash, 64 bits). Images whose hashes differ
# in <= this many bits are treated as near duplicates and kept in the SAME split.
NEAR_DUPLICATE_MAX_HAMMING = 4

# --------------------------------------------------------------------------- #
# Augmentation (moderate: keeps the drape recognisable)
# --------------------------------------------------------------------------- #
# NOTE: a horizontal flip mirrors the side on which the pallu falls. Set to False
# if you think pallu side is part of what defines the style in your data.
USE_HORIZONTAL_FLIP = True
AUG_ROTATION = 0.03        # fraction of a full turn  (0.03 -> about +-11 degrees)
AUG_ZOOM = 0.10            # +-10 %
AUG_TRANSLATION = 0.08     # +-8 % of width/height
AUG_BRIGHTNESS = 0.15      # +-15 % (pixel range 0..1)

# --------------------------------------------------------------------------- #
# Prediction thresholds
# --------------------------------------------------------------------------- #
# Supervised binary mode: p(Marathi) >= CLASSIFICATION_THRESHOLD -> Marathi.
CLASSIFICATION_THRESHOLD = 0.5
# Supervised binary mode: confidence = max(p, 1-p). Below this -> "UNCERTAIN".
CONFIDENCE_THRESHOLD = 0.70

# Positive-only mode: threshold = this percentile of cosine distances measured
# on held-out (validation) Marathi images.
ONE_CLASS_PERCENTILE = 95
# Positive-only mode: distances within +-this fraction of the threshold -> "UNCERTAIN".
ONE_CLASS_UNCERTAIN_MARGIN = 0.10


def ensure_directories() -> None:
    """Create the output folders if they do not exist."""
    for folder in (DATASET_DIR, POSITIVE_DIR, NEGATIVE_DIR, MODELS_DIR,
                   OUTPUTS_DIR, TEST_IMAGES_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Seed Python, NumPy and TensorFlow (imported lazily)."""
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)  # seeds python, numpy and tf together
    except ImportError:
        pass
