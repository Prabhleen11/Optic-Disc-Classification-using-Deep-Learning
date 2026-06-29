"""
augmentation.py

Builds Keras ImageDataGenerators for training (with augmentation) and
validation (no augmentation, just rescaling) from a directory structure of:
    data_dir/train/<class_name>/*.jpg
    data_dir/val/<class_name>/*.jpg
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from data_preprocessing import IMG_SIZE

BATCH_SIZE = 16  # smaller default batch size, common for limited medical imaging datasets


def get_data_generators(data_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=(0.85, 1.15),
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",  # Normal vs Glaucoma; switch to "categorical" for multi-class
        shuffle=True,
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )

    return train_generator, val_generator
