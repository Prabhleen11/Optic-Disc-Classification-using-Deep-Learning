"""
data_preprocessing.py

Preprocesses retinal fundus images: resizing, normalization, and CLAHE-based
contrast enhancement to make optic disc/cup boundaries more distinguishable.
"""

import cv2
import numpy as np

IMG_SIZE = (224, 224)


def apply_clahe(image_bgr, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) on the
    L channel of the LAB color space, which improves local contrast without
    over-amplifying noise — useful for highlighting optic disc/cup edges.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((l_enhanced, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_bgr


def preprocess_image(image_path, img_size=IMG_SIZE, apply_contrast_enhancement=True):
    """
    Loads an image from disk, applies CLAHE (optional), resizes, and
    normalizes pixel values to [0, 1]. Returns a numpy array ready for
    model input (without batch dimension).
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")

    if apply_contrast_enhancement:
        image = apply_clahe(image)

    image = cv2.resize(image, img_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype("float32") / 255.0
    return image


def preprocess_for_inference(image_path, img_size=IMG_SIZE):
    """Same as preprocess_image, but adds the batch dimension for model.predict()."""
    image = preprocess_image(image_path, img_size=img_size)
    return np.expand_dims(image, axis=0)
