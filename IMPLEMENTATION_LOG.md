# Implementation Log — ASL Hand Tracking Translator

## v0.1 — Project Setup and Hand Detector
**Date:** 2025-11-27

### Objectives
- Initialize project folder structure
- Set up Python environment with required libraries
- Implement baseline hand detection using MediaPipe

### Tasks Completed
- Making sure that I use a Python <= 3.11
- Installed required packages: `opencv-python`, `mediapipe`, `numpy`, `scikit-learn`, `pandas`, `matplotlib` and `pyspellchecker`
- Created `hand_detector.py` class:
  - Detect hands in real-time from webcam feed
  - Extract hand landmarks (x, y, z coordinates)
  - Draw landmarks and connections on the video feed
- Verified webcam integration and FPS display


## v0.2 — Start to Collect the Data  
**Date:** 2025-11-28  

### Objectives
- Collect hand landmark data for ASL letters.  
- Save the landmarks in `.npy` format for later model training.  
- Implement flexible collection with preparation time and controlled delays between samples.

### Tasks Completed
- Created `data_collection.py` script with the following features:
  - Prompts user for which ASL letter to collect.
  - Initializes webcam and MediaPipe hand tracking.
  - Adds a **5-second preparation timer** so the user can get into position.
  - Adds a **2-second delay between saved samples** to avoid identical frames.
  - Captures a configurable number of samples (`NUM_SAMPLES`).
  - Saves each sample to a `.npy` file containing flattened `(x, y, z)` landmark coordinates.
- Added **resume support**:
  - If files already exist for the target letter, the script continues collecting from the last index.
- Successfully collected and verified a dataset for letter **A**.
- Learned how to flatten 21–landmark (x, y, z) data into NumPy arrays.

### New Realizations
#### 1. J and Z Cannot Be Collected Like the Other Letters  
ASL letters **J** and **Z** involve **movement**, not static hand poses.  
This means:
- They cannot be captured using single-frame landmark snapshots.  
- They will require a different collection system that records sequences (video-based landmarks).  
- They should be excluded from the initial static dataset.

#### 2. Dataset Progress (Completed Letters)
I have finished collecting dataset samples for the following ASL letters:

**A, B, C, D, E, F, G, H, I, K, L, M, N, S, T, U, V**

The goal is ~250 samples per letter, and currently these letters have completed sets.

### Notes & Challenges
- Some samples were unusable due to:
  - The hand moving too fast,
  - The camera missing landmarks,
  - Double-detected or partially detected hands.
- These bad samples will be removed or filtered out during the preprocessing stage.
- Collecting too many samples at once makes the hand tired — working in shorter sessions produces better results.
- Stable hand positioning improves MediaPipe detection accuracy significantly.


## v0.3 — Dataset Completion
**Date:** 2025-12-13

### Summary
- Completed data collection for **all ASL letters (A–Z)**.
- Static letters collected as **single-frame landmark samples**.
- Dynamic letters **J** and **Z** collected as **landmark sequences**.

### Work Done
- Collected ~**250 samples per letter** for a balanced dataset.
- Used a separate sequence-based script for J and Z with on-screen countdown.
- Verified all data is saved correctly in `.npy` format without overwriting.

### Notes
- Movement letters required more retries due to motion variability.
- Countdown before recording improved consistency and starting position.
- Dataset includes natural variation, not perfectly clean data.

### Next Step
- Start **data preprocessing and cleanup** before model training.


## v0.4 — Data Processing Pipeline
**Date:** 2025-12-15

### Objectives
- Implement a complete data preprocessing pipeline.
- Prepare static and movement datasets for machine learning models.

### Work Done
- Created `data_processing.py` to handle all dataset preprocessing tasks.
- Implemented landmark normalization:
  - Wrist landmark used as the origin.
  - Landmarks scaled by maximum distance for scale invariance.
- Added validation checks to filter out invalid or corrupted samples.
- Processed **static letters (A–I, K–Y)**:
  - Loaded single-frame landmark samples.
  - Encoded letter labels as integers.
  - Split data into training (70%), validation (15%), and test (15%) sets.
- Processed **movement letters (J and Z)** separately:
  - Loaded full landmark sequences.
  - Normalized each frame individually.
  - Kept J and Z as independent datasets due to different sequence lengths.
  - Split each movement dataset into training, validation, and test sets.
- Saved all processed datasets into the `data/processed/` directory in `.npy` format.

### Notes
- Static and movement data are intentionally processed separately to simplify
  model design and avoid sequence padding or trimming.
- This pipeline ensures balanced class representation across all dataset splits.

### Next Step
- Begin **model selection and training** using the processed datasets.


## v0.5 — Static Model Selection
**Date:** 2025-12-17

### Objectives
- Train and compare models for static ASL letters.
- Select the best-performing model.

### Work Done
- Created `choose_static_model.py` with `StaticModelSelector` class.
- Trained MLP, Random Forest, Gradient Boosting, and XGBoost on processed static data.
- Compared models using validation accuracy and Matplotlib visualization.

### Results
- MLP: 99.47% 
- Random Forest: 99.33%
- Gradient Boosting: 99.07%
- XGBoost: 98.93%

**Selected Model:** MLP (`models/static_best_model_MLP.joblib`) for future inference.

### Next Step
- Start model selection for movement-based letters (J and Z).


## v0.6 — Movement Model Selection & Real-World Considerations
**Date:** 2025-12-20

### Objectives
- Train and compare models for movement-based ASL letters (J and Z).
- Select the best-performing model per letter.
- Acknowledge real-world performance limitations.

### Work Done
- Created `choose_movement_model.py` with `MovementModelSelector` class.
- Trained multiple sequence models on processed movement data:
  - LSTM, GRU, 1D-CNN, Bidirectional LSTM, Stacked LSTM.
- Evaluated models using validation accuracy.
- Saved the best-performing model for each letter:
  - **Letter J:** LSTM (`models/movement_J_best_LSTM.keras`)
  - **Letter Z:** LSTM (`models/movement_Z_best_LSTM.keras`)
- Added Matplotlib visualization of model comparisons for each letter (`choose_movement_model_j.png`, `choose_movement_model_z.png`).

### Notes
- Achieved **100% validation accuracy** on the processed dataset for both J and Z.
- Real-world performance may **not reach 100%** due to:
  - Variations in lighting, background, hand size, and camera angles.
  - MediaPipe tracking errors or missed landmarks.
  - Differences in motion speed and hand positioning.
- Future steps will include testing with **real webcam input and pre-recorded videos** to evaluate true performance.

### Next Step
- Integrate static and movement models into a **single inference pipeline**.
- Begin **real-time prediction and recording** scripts to generate test sentences.
- Prepare for **NLP/LLM integration** for error correction and sentence reconstruction.