"""Evaluate on the held-out TEST split (same split that train.py created).

Usage:
    python src/evaluate.py
    python src/evaluate.py --split val
    python src/evaluate.py --extra-negatives path/to/folder_of_non_marathi_images
        (positive-only mode: measure rejection on images NOT used in training)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src import config
from src.dataset import DatasetError, load_images, load_splits_from_manifest
from src.preprocessing import discover_images

import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.model import (  # noqa: E402
    cosine_distance,
    load_classifier,
    load_feature_extractor,
    load_one_class_stats,
    load_training_mode,
)

CLASS_NAMES = ["Non-Marathi", "Marathi"]


def plot_confusion_matrix(cm: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report_binary(y_true, y_pred, y_score, title: str) -> str:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())

    lines = [
        "=" * 60, title, "=" * 60,
        f"Samples   : {len(y_true)} (Marathi={int((y_true == 1).sum())}, "
        f"Non-Marathi={int((y_true == 0).sum())})",
        f"Accuracy  : {accuracy_score(y_true, y_pred):.4f}",
        f"Precision : {precision_score(y_true, y_pred, zero_division=0):.4f}",
        f"Recall    : {recall_score(y_true, y_pred, zero_division=0):.4f}",
        f"F1-score  : {f1_score(y_true, y_pred, zero_division=0):.4f}",
    ]
    if y_score is not None and len(set(y_true.tolist())) == 2:
        lines.append(f"ROC AUC   : {roc_auc_score(y_true, y_score):.4f}")
    lines += [
        "", "Confusion matrix (rows=actual, cols=predicted; order: Non-Marathi, Marathi)",
        str(cm), "",
        f"True Positives  (Marathi predicted Marathi)          : {tp}",
        f"True Negatives  (Non-Marathi predicted Non-Marathi)  : {tn}",
        f"False Positives (Non-Marathi predicted Marathi)      : {fp}",
        f"False Negatives (Marathi predicted Non-Marathi)      : {fn}",
        "", "Classification report:",
        classification_report(y_true, y_pred, labels=[0, 1], target_names=CLASS_NAMES,
                              zero_division=0),
    ]
    text = "\n".join(lines)
    print(text)
    plot_confusion_matrix(cm, config.OUTPUTS_DIR / "confusion_matrix.png", title)
    print(f"Saved plot: {config.OUTPUTS_DIR / 'confusion_matrix.png'}")
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the trained model.")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--extra-negatives", type=Path, default=None,
                        help="folder of NON-Marathi images not used in training "
                             "(positive-only mode only)")
    args = parser.parse_args()

    config.ensure_directories()
    config.set_global_seed()
    try:
        mode = load_training_mode()["mode"]
        data = load_splits_from_manifest()
    except (FileNotFoundError, DatasetError) as exc:
        print(f"ERROR: {exc}")
        return 1

    X = data.X_test if args.split == "test" else data.X_val
    y = data.y_test if args.split == "test" else data.y_val
    if len(X) == 0:
        print("ERROR: the evaluation split is empty.")
        return 1

    if mode == "binary":
        model = load_classifier()
        prob = model.predict(X, batch_size=config.BATCH_SIZE, verbose=0).ravel()
        pred = (prob >= config.CLASSIFICATION_THRESHOLD).astype(int)
        text = report_binary(y, pred, prob,
                             f"SUPERVISED BINARY EVALUATION ({args.split} split)")
        text += ("\n\nNote: these numbers come from a small held-out split of YOUR dataset.\n"
                 "They do not guarantee the same performance on arbitrary outside images.")
    else:
        extractor = load_feature_extractor()
        stats = load_one_class_stats()
        feats = extractor.predict(X, batch_size=config.BATCH_SIZE, verbose=0)
        dist_pos = cosine_distance(feats, stats["centroid"])
        accepted = dist_pos <= stats["threshold"]
        lines = [
            "=" * 60,
            f"POSITIVE-ONLY EVALUATION ({args.split} split)",
            "=" * 60,
            "WARNING: only Marathi images were used. False-positive rate CANNOT be",
            "measured without non-Marathi images.",
            f"Distance threshold           : {stats['threshold']:.4f}",
            f"Held-out Marathi images      : {len(dist_pos)}",
            f"Accepted as LIKELY MARATHI   : {int(accepted.sum())} "
            f"({100.0 * accepted.mean():.1f}%)   <- recall on Marathi only",
        ]
        print("\n".join(lines))
        text = "\n".join(lines)
        if args.extra_negatives is not None:
            neg_paths = discover_images(args.extra_negatives)
            if not neg_paths:
                print(f"ERROR: no images found in {args.extra_negatives}")
                return 1
            X_neg, kept, failed = load_images(neg_paths)
            print(f"\nLoaded {len(kept)} external negatives ({len(failed)} unreadable).")
            dist_neg = cosine_distance(
                extractor.predict(X_neg, batch_size=config.BATCH_SIZE, verbose=0),
                stats["centroid"])
            y_true = np.concatenate([np.ones(len(dist_pos)), np.zeros(len(dist_neg))])
            dist_all = np.concatenate([dist_pos, dist_neg])
            y_pred = (dist_all <= stats["threshold"]).astype(int)
            text += "\n\n" + report_binary(y_true, y_pred, -dist_all,
                                           "POSITIVE-ONLY + EXTERNAL NEGATIVES")
        else:
            print("\nTip: pass --extra-negatives <folder> to measure how many "
                  "non-Marathi images are wrongly accepted.")

    report_path = config.OUTPUTS_DIR / "evaluation_report.txt"
    report_path.write_text(text, encoding="utf-8")
    print(f"Saved report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
