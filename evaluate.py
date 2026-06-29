"""
evaluate.py

Evaluates a trained model on the validation/test set: accuracy, precision,
recall, AUC-ROC, and a confusion matrix. AUC-ROC is emphasized since medical
imaging datasets are commonly imbalanced (more Normal than Glaucoma cases).

Usage:
    python evaluate.py --data_dir dataset --model_path saved_model/optic_disc_model.h5
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
)
from tensorflow.keras.models import load_model

from augmentation import get_data_generators
from data_preprocessing import IMG_SIZE


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate optic disc classification model")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()

    _train_gen, val_gen = get_data_generators(args.data_dir, img_size=IMG_SIZE, batch_size=args.batch_size)
    model = load_model(args.model_path)

    val_gen.reset()
    y_true = val_gen.classes
    y_pred_probs = model.predict(val_gen, steps=len(val_gen)).flatten()
    y_pred = (y_pred_probs >= 0.5).astype(int)

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=list(val_gen.class_indices.keys())))

    auc = roc_auc_score(y_true, y_pred_probs)
    print(f"AUC-ROC: {auc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve.png")
    print("Saved ROC curve to roc_curve.png")


if __name__ == "__main__":
    main()
