# ASL Hand Tracking Translator
![Python](https://img.shields.io/badge/Python-3.x-blue)

... TO DO ...

_Created by **Réka Gábosi**_

## Table of Contents

... TO DO ...

## Description

A computer vision and machine learning project that recognizes American Sign Language (ASL) letters
using real-time hand landmark tracking and custom-trained models.

### Hand Landmark Representation

The project uses **MediaPipe Hand Tracking**, which represents each hand using
**21 key landmarks**, each with `(x, y, z)` coordinates.

![Hand Landmarks](images/Hand_Landmarks.png)

## File Structure

```bash
asl-hand-tracking-translator/
│
├── data/                           # All dataset-related files
│   ├── processed/                  # Cleaned and model-ready data
│   │   ├── static_X_train.npy
│   │   ├── static_y_train.npy
│   │   ├── static_X_val.npy
│   │   ├── static_y_val.npy
│   │   ├── static_X_test.npy
│   │   ├── static_y_test.npy
│   │   ├── movement_J_X_train.npy
│   │   ├── movement_J_y_train.npy
│   │   ├── movement_J_X_val.npy
│   │   ├── movement_J_y_val.npy
│   │   ├── movement_J_X_test.npy
│   │   ├── movement_J_y_test.npy
│   │   ├── movement_Z_X_train.npy
│   │   ├── movement_Z_y_train.npy
│   │   ├── movement_Z_X_val.npy
│   │   ├── movement_Z_y_val.npy
│   │   ├── movement_Z_X_test.npy
│   │   └── movement_Z_y_test.npy
│   │
│   ├── raw_landmarks/              # Static ASL letters (single-frame landmarks)
│   │   ├── A/                      # One folder per letter
│   │   │   ├── A_0.npy             # One sample = one hand pose
│   │   │   ├── A_1.npy
│   │   │   └── ...
│   │   ├── B/
│   │   ├── C/
│   │   ├── ...
│   │   └── Y/
│   │
│   └── movement_sequences/         # Dynamic ASL letters (movement-based)
│       ├── J/                      # One folder per letter
│       │   ├── J_0.npy             # One file = one full movement sequence
│       │   ├── J_1.npy
│       │   └── ...
│       └── Z/
│           ├── Z_0.npy
│           ├── Z_1.npy
│           └── ...
│
├── models/
│   └── static_best_model_MLP.joblib # Best-performing static letter model
│    
├── src/                            # Source code for the project
│   ├── hand_detector.py            # MediaPipe hand detection and landmark extraction
│   ├── data_collection.py          # Script for collecting static letter samples
│   ├── data_processing.py          # Data cleaning, normalization, splitting
│   ├── choose_static_model.py      # Trains and compares static ASL models
│   └── movement_data_collection.py # Script for collecting J and Z movement sequences
│
├── images/                         # Images used for documentation
│   ├── Hands_Landmarks.png         # Hand landmark reference image
│   └── choose_static_model.png     # Static model comparison results
│
├── test/                           # Testing and validation utilities
│   ├── test_data/                  # Small test dataset
│   │   └── A/
│   │       ├── A_0.npy
│   │       ├── A_1.npy
│   │       └── ...
│   └── test_data_collection.py     # Script to test data collection logic
│
├── requirements.txt                # Python dependencies
├── IMPLEMENTATION_LOG.md           # Development progress and version history             
├── README.md                       # Project overview     
└── LICENSE CC BY-ND 4.0            # Project license
```

## Dataset

The dataset used in this project was **collected manually** using a webcam and **MediaPipe hand tracking**.

Each sample consists of:
- **21 hand landmarks**
- **3 coordinates per landmark** (`x`, `y`, `z`)

### Dataset Types

- **Static dataset**  
  Single-frame landmark samples representing static ASL letters.

- **Movement dataset**  
  Multi-frame landmark sequences representing dynamic ASL letters (**J** and **Z**).

This approach avoids using pre-built datasets and ensures full control over
data quality, consistency, and preprocessing.

### Dataset Size

The dataset contains **approximately 250 samples per ASL letter**, including both
static and movement-based letters.

- **Static letters (A–I, K–Y):**
  - ~250 single-frame landmark samples per letter

- **Movement letters (J, Z):**
  - ~250 movement sequences per letter  
  - J sequences contain 40 frames each  
  - Z sequences contain 60 frames each  

> **Note:**  
> For movement-based letters, a larger number of samples would typically improve
> model robustness due to higher variability in motion.  
> However, for this portfolio project, the dataset was intentionally kept consistent
> at ~250 samples per letter to evaluate how well the model performs under limited
> movement data conditions.

## Data Processing

### Static letters (A-I, K-Y)

Static ASL letters are represented as **single-frame hand poses**.

**Processing steps:**
- Load raw `.npy` landmark files (63 values per sample).
- Normalize landmarks:
  - Wrist is used as the origin.
  - All landmarks are scaled by maximum distance from the wrist.
- Filter out invalid or corrupted samples.
- Encode letter labels as integers.
- Split data into:
  - Training set (70%)
  - Validation set (15%)
  - Test set (15%)
- Save processed datasets into the `data/processed/` folder.

**Output files:**
- `static_X_train.npy`
- `static_y_train.npy`
- `static_X_val.npy`
- `static_y_val.npy`
- `static_X_test.npy`
- `static_y_test.npy`

### Movement Letters (J and Z)

ASL letters **J** and **Z** involve motion and are handled separately from static letters.

Each movement sample is stored as a **sequence of frames**:
- J → 40 frames per sequence
- Z → 60 frames per sequence

**Processing steps:**
- Load full movement sequences from disk.
- Validate sequence length and feature shape.
- Normalize each frame individually using the same wrist-based method.
- Keep J and Z as **separate datasets** due to different sequence lengths.
- Split each letter into training, validation, and test sets.
- Save processed movement datasets separately.

**Output files:**
- `movement_J_X_train.npy`
- `movement_J_y_train.npy`
- `movement_J_X_val.npy`
- `movement_J_y_val.npy`
- `movement_J_X_test.npy`
- `movement_J_y_test.npy`

- `movement_Z_X_train.npy`
- `movement_Z_y_train.npy`
- `movement_Z_X_val.npy`
- `movement_Z_y_val.npy`
- `movement_Z_X_test.npy`
- `movement_Z_y_test.npy`

## Model Selection and Training (Static Letters)

To identify the most suitable model for recognizing **static ASL letters**, multiple
machine learning classifiers were trained and evaluated using the same processed dataset.

All models were trained on the **training set** and evaluated on the **validation set**
to ensure a fair comparison.

### Models evaluated
- Multi-Layer Perceptron (MLP)
- Random Forest
- Gradient Boosting
- XGBoost

### Validation Results

| Model               | Validation Accuracy |
|--------------------|---------------------|
| MLP                | **99.47%**          |
| Random Forest      | 99.33%              |
| Gradient Boosting  | 99.07%              |
| XGBoost            | 98.93%              |

The following figure shows a visual comparison of validation accuracy across models:

![Static Model Comparison](images/choose_static_model.png)

### Best Performing Model

Based on validation accuracy, the **MLP classifier** achieved the best performance
and was selected as the final static-letter recognition model.

**Selected model:**
- **Model:** Multi-Layer Perceptron (MLP)
- **Validation accuracy:** **99.47%**

The best-performing model was saved for later use in real-time inference and integration
with movement-based models and NLP post-processing.