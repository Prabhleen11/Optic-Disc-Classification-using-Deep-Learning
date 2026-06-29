"""
predict.py

Loads a trained model and predicts whether a single retinal fundus image
is Normal or Glaucoma-suspect.

Usage:
    python predict.py --image_path sample_fundus.jpg --model_path saved_model/optic_disc_model.h5
"""

import argparse
import json
import os

from tensorflow.keras.models import load_model

from data_preprocessing import preprocess_for_inference, IMG_SIZE


def parse_args():
    parser = argparse.ArgumentParser(description="Predict optic disc classification for a single image")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--class_indices_path", type=str, default="saved_model/class_indices.json",
        help="JSON file mapping class names to indices (saved during training)"
    )
    return parser.parse_args()


def load_class_names(class_indices_path):
    if os.path.exists(class_indices_path):
        with open(class_indices_path, "r") as f:
            class_indices = json.load(f)
        return {v: k for k, v in class_indices.items()}
    # Fallback default mapping if file not found
    return {0: "Normal", 1: "Glaucoma"}


def main():
    args = parse_args()

    model = load_model(args.model_path)
    img_array = preprocess_for_inference(args.image_path, img_size=IMG_SIZE)

    prob = float(model.predict(img_array)[0][0])
    class_names = load_class_names(args.class_indices_path)

    predicted_index = 1 if prob >= 0.5 else 0
    predicted_class = class_names.get(predicted_index, str(predicted_index))
    confidence = prob if predicted_index == 1 else 1 - prob

    print(f"Predicted Class : {predicted_class}")
    print(f"Confidence      : {confidence * 100:.2f}%")
    print("\nDisclaimer: This is a prototype model for educational/portfolio purposes "
          "only and is not a validated diagnostic tool.")


if __name__ == "__main__":
    main()
