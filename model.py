"""
model.py

Defines the classification model for optic disc / glaucoma screening.
Supports two modes:
  - "custom": a CNN built from scratch (good for understanding fundamentals)
  - "resnet50": transfer learning using a pretrained ResNet50 backbone
                (better performance on small medical imaging datasets)
"""

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization,
    GlobalAveragePooling2D, Input,
)
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam


def build_custom_cnn(input_shape=(224, 224, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),  # binary: Normal vs Glaucoma-suspect
    ])
    return model


def build_resnet50_model(input_shape=(224, 224, 3), freeze_backbone=True):
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)

    if freeze_backbone:
        base_model.trainable = False

    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=not freeze_backbone)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.4)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    return model


def build_model(input_shape=(224, 224, 3), backbone="custom", learning_rate=1e-3):
    """
    backbone: "custom" or "resnet50"
    """
    if backbone == "resnet50":
        model = build_resnet50_model(input_shape=input_shape)
    else:
        model = build_custom_cnn(input_shape=input_shape)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", "AUC", "Precision", "Recall"],
    )
    return model
