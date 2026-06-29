"""
train.py

Trains the optic disc classification model and saves the best checkpoint
based on validation AUC (more meaningful than accuracy for imbalanced
medical imaging datasets).

Usage:
    python train.py --data_dir dataset --epochs 30 --backbone custom
    python train.py --data_dir dataset --epochs 30 --backbone resnet50
"""

import argparse
import os
import json

import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from augmentation import get_data_generators
from data_preprocessing import IMG_SIZE
from model import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train optic disc classification model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--backbone", type=str, default="custom", choices=["custom", "resnet50"])
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="saved_model")
    return parser.parse_args()


def plot_history(history, output_dir):
    metrics = ["accuracy", "loss", "auc"]
    available = [m for m in metrics if m in history.history]

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4))
    if len(available) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available):
        ax.plot(history.history[metric], label=f"train_{metric}")
        val_key = f"val_{metric}"
        if val_key in history.history:
            ax.plot(history.history[val_key], label=f"val_{metric}")
        ax.set_title(metric.capitalize())
        ax.set_xlabel("Epoch")
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"))
    print(f"Saved training history plot to {output_dir}/training_history.png")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train_gen, val_gen = get_data_generators(args.data_dir, img_size=IMG_SIZE, batch_size=args.batch_size)

    # Save class index mapping for later use during inference
    with open(os.path.join(args.output_dir, "class_indices.json"), "w") as f:
        json.dump(train_gen.class_indices, f)
    print(f"Class indices: {train_gen.class_indices}")

    model = build_model(
        input_shape=IMG_SIZE + (3,),
        backbone=args.backbone,
        learning_rate=args.learning_rate,
    )
    model.summary()

    checkpoint_path = os.path.join(args.output_dir, "optic_disc_model.h5")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        ModelCheckpoint(checkpoint_path, monitor="val_auc", mode="max", save_best_only=True),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    plot_history(history, args.output_dir)
    print(f"Best model saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
