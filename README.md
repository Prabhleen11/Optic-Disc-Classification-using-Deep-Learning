# Optic-Disc-Classification-using-Deep-Learning
A deep learning pipeline for classifying retinal fundus images to screen for glaucoma-related optic disc abnormalities. Includes CLAHE-based contrast enhancement, data augmentation, and a CNN/ResNet50 model with AUC-ROC evaluation. Built with Python, TensorFlow/Keras, and OpenCV.
[README.md](https://github.com/user-attachments/files/29455273/README.md)

## Overview

A deep learning pipeline that classifies retinal fundus images based on
optic disc appearance, for early-stage screening of glaucoma-related
abnormalities. The goal is to explore whether a CNN can learn to flag
visually suspicious optic discs (e.g. abnormal cup-to-disc ratio,
irregular rim shape) from a labeled retinal image dataset, as a
proof-of-concept for an automated screening aid.

## Problem Statement

Glaucoma is a leading cause of irreversible blindness, and early detection
through regular eye screening significantly reduces vision loss. Manual
screening requires a trained ophthalmologist examining the optic disc in a
fundus image, which isn't always quickly accessible, especially in
under-resourced areas. This project investigates whether a CNN-based
classifier can serve as a first-pass screening tool to flag images that
warrant closer review.

## Approach

1. **Data preprocessing** — Fundus images are resized to a fixed
   resolution, normalized, and enhanced with contrast adjustment (CLAHE)
   to make the optic disc and cup boundaries more distinguishable.
2. **Data augmentation** — Rotation, flipping, and brightness/contrast
   jitter are applied to artificially expand the training set and improve
   generalization, given that medical imaging datasets are often small.
3. **Model architecture** — A CNN built with Keras/TensorFlow (with an
   optional transfer-learning variant using a pretrained ImageNet backbone
   such as ResNet50/EfficientNet, fine-tuned on the retinal dataset).
4. **Training & evaluation** — Trained with binary cross-entropy loss
   (Normal vs. Glaucoma-suspect) and tracked accuracy, precision, recall,
   and AUC-ROC, since class imbalance is common in medical datasets and
   accuracy alone can be misleading.
5. **Hyperparameter tuning** — Learning rate, dropout, and augmentation
   intensity were tuned based on validation loss to reduce overfitting,
   using early stopping and learning-rate reduction on plateau.

## Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow / Keras
- **Image Processing:** OpenCV
- **Data Handling:** NumPy, Pandas
- **Visualization:** Matplotlib
- **Environment:** Kaggle Notebooks

## Project Structure

```
optic-disc-classification/
├── README.md
├── requirements.txt
├── data_preprocessing.py   # Resizing, normalization, CLAHE contrast enhancement
├── augmentation.py         # Image augmentation pipeline
├── model.py                # CNN architecture (custom + transfer-learning option)
├── train.py                # Training script with callbacks and metric tracking
├── evaluate.py             # Evaluation: accuracy, precision, recall, AUC-ROC, confusion matrix
└── predict.py               # Inference on a single retinal image
```

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Organize your dataset as:
#    dataset/
#      train/Normal/*.jpg
#      train/Glaucoma/*.jpg
#      val/Normal/*.jpg
#      val/Glaucoma/*.jpg

# 3. Train the model
python train.py --data_dir dataset --epochs 30 --backbone custom
# or, using transfer learning:
python train.py --data_dir dataset --epochs 30 --backbone resnet50

# 4. Evaluate on the validation/test set
python evaluate.py --data_dir dataset --model_path saved_model/optic_disc_model.h5

# 5. Run inference on a single image
python predict.py --image_path sample_fundus.jpg --model_path saved_model/optic_disc_model.h5
```

## Notes on the Dataset

This pipeline is designed to work with publicly available retinal fundus
image datasets used for glaucoma screening research (e.g. ORIGA, RIM-ONE,
or Kaggle's glaucoma detection datasets). It expects a binary folder
structure (`Normal/` vs `Glaucoma/`) but can be adapted for multi-class
labels (e.g. including "Suspect" as a third class) by adjusting `model.py`
and `train.py`.

## Results

- Achieved meaningful separation between normal and glaucoma-suspect classes
  on the validation set, with AUC-ROC used as the primary metric due to
  class imbalance in the dataset.
- Augmentation and contrast enhancement noticeably reduced overfitting
  compared to training on raw, unaugmented images.

## Important Disclaimer

This project is a learning/portfolio prototype, **not a validated medical
diagnostic tool**. It has not been clinically validated and should not be
used for real diagnosis or treatment decisions. Any real-world deployment
would require regulatory approval, much larger and more diverse datasets,
and validation by medical professionals.

## Future Improvements

- Use Grad-CAM to visualize which regions of the optic disc the model
  focuses on, for interpretability and clinician trust.
- Expand to multi-class severity grading instead of binary classification.
- Validate across multiple public datasets to test generalization across
  different camera types and populations.
