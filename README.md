# Food-101 Image Classifier
### Transfer Learning with EfficientNetB0 | Adversarial Robustness Research

---

## Overview

A deep learning model that classifies food images across 101 categories using transfer learning on the Food-101 benchmark dataset. Built as an exploration of ML model behavior, input sensitivity, and data integrity. Focusing on understanding where and how image classifiers can be manipulated or deceived.

This project is part of an ongoing transition into **AI/ML security engineering**, using a practical classification task as a foundation for studying adversarial robustness concepts.

---

## Background

This project started as an independent rebuild after a Coursera ML course, my attempt to apply transfer learning without following a tutorial step by step. I wanted to build my own ml where i could count calories in my food based on the image. The original version had a broken training pipeline (model architecture commented out, hardcoded Windows paths, no proper train/test split). Debugging and completing it became as valuable as building it.

This project started after a Coursera ML course, but instead of following along, I wanted to build something I'd actually use: a calorie counter that could identify food from a photo and estimate its nutritional content. No tutorial, no guided steps, just an idea and a dataset.

The original version had a broken training pipeline (model architecture commented out, hardcoded Windows paths, no proper train/test split). Debugging and completing it became as valuable as building it.

Tracing the 1% accuracy, resolving Keras version conflicts, working through Windows long path errors, diagnosing a Phase 2 training collapse has taught me more about how ML pipelines fail than building it correctly the first time would have. That debugging process is now part of why this project is relevant to security engineering: understanding how systems break is the job.

---
## What It Does

- Classifies food images into 101 categories
- Returns top-5 predictions per image with a visual confidence breakdown
- Demonstrates how small input disturbances (brightness, contrast, rotation) affect model confidence

---

## Model Architecture

| Component | Detail |
|---|---|
| Base model | EfficientNetB0 (pretrained on ImageNet) |
| Training strategy | Two-phase transfer learning |
| Phase 1 | Head-only training, base frozen (LR: 1e-3) |
| Phase 2 | Fine-tuning top 20 layers (LR: 1e-5) |
| Input size | 224 × 224 × 3 |
| Output | 101-class softmax |
| Dataset | Food-101 (75,750 train / 25,250 test) |

**EfficientNetB0?** EfficientNet models are designed to balance accuracy and algorithmic efficiency. The ImageNet pre-training means the model already understands low-level visual features (edges, textures, color patterns) before seeing a food image.

---

## Security & Robustness Notes

This project was developed with an eye toward AI security concepts:

**Data Integrity**
The pipeline includes an image verification step that scans all training images for corruption before training begins. In a real-world deployment, corrupt training data is a known attack vector this is a basic but important defense.

**Input Sensitivity**
The augmentation pipeline applies brightness shifts, contrast changes, rotations, and different zooms during training. These transformations mirror the kinds of disruptions used in adversarial example research (e.g. FGSM, patch attacks). Training on augmented data improves robustness to naturally-occurring distribution shifts.

**Distribution Shift**
The gap between training accuracy and test accuracy is a practical example of distribution shift, a model that overfits to training data fails in the real world. Understanding this failure mode is foundational to AI security testing.

---

## Project Structure

```
food-101-classifier/
├── Calorie_Counter_v2.ipynb   # Main notebook
├── food101_efficientnet.h5    # Saved model weights
├── training_history.png       # Accuracy/loss curves
├── food-101/
│   ├── images/                # Raw dataset (101 classes)
│   ├── train/                 # Official training split (75,750 images)
│   ├── test/                  # Official test split (25,250 images)
│   └── meta/                  # train.txt / test.txt split files
└── README.md
```

---

## Setup & Usage

**Requirements**
```bash
pip install tensorflow pillow scikit-learn matplotlib numpy
```

**Run the notebook**

Open `Calorie_Counter_v2.ipynb` in Jupyter or VS Code and run cells top to bottom. The dataset downloads automatically on first run (~5GB).

**Classify a new image**

Load the saved model and run the inference cell:
```python
# In Cell 13 — uncomment to load saved model
model = tf.keras.models.load_model('food101_efficientnet.h5')

# In Cell 12 — update this path
TEST_IMAGE_PATH = 'path/to/your/image.jpg'
display_prediction(TEST_IMAGE_PATH)
```

---

## Training Details

Training runs in two phases to avoid destroying pretrained ImageNet weights:

**Phase 1 — Feature Extraction (base frozen)**
Only the classification head trains. Fast convergence, safe starting point.

**Phase 2 — Fine-Tuning (last 20 layers unfrozen)**
The top layers of EfficientNetB0 adapt to food-specific features at a very low learning rate (1e-5). Going too high here destroys the pretrained weights.

Callbacks used:
- `ModelCheckpoint` — saves best model automatically based on val_accuracy
- `EarlyStopping` — stops training if validation accuracy plateaus
- `ReduceLROnPlateau` — halves learning rate when loss stalls

---

## What I Learned

- How transfer learning actually works under the hood, not just what it is, but why freezing/unfreezing layers matters and what happens when you get it wrong
- How data pipeline decisions (batch size, augmentation, official vs. random splits) affect results and dependanility
- How to debug a broken training pipeline: the original notebook had the model architecture commented out, which caused ~1% accuracy across all runs


---

## Next Steps

- [ ] Fix class index alignment between train and test generators
- [ ] Add FGSM adversarial example generation
- [ ] Evaluate model accuracy on perturbed vs. clean inputs
- [ ] Wrap inference cell as a REST API endpoint (FastAPI)
- [ ] Add calorie estimation via Nutritionix API post-classification

---

## References

- [Food-101 Dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) — Bossard et al., 2014
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946) — Tan & Le, 2019
- [TensorFlow Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
