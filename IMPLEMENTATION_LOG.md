# Implementation Log — ASL Hand Tracking Translator

## v0.1 — Project Setup and Hand Detector

**Date:** 2025-11-27

**Objectives:**
- Initialize project folder structure
- Set up Python environment with required libraries
- Implement baseline hand detection using MediaPipe

**Tasks Completed:**
- Making sure that I use a Python <= 3.11
- Installed required packages: `opencv-python`, `mediapipe`, `numpy`, `scikit-learn`, `pandas`, `matplotlib` and `pyspellchecker`
- Created `hand_detector.py` class:
  - Detect hands in real-time from webcam feed
  - Extract hand landmarks (x, y, z coordinates)
  - Draw landmarks and connections on the video feed
- Verified webcam integration and FPS display

# v0.2 — Start to Collect the Data  
**Date:** 2025-11-28  

## Objectives
- Collect hand landmark data for ASL letters.  
- Save the landmarks in `.npy` format for later model training.  
- Implement flexible collection with preparation time and controlled delays between samples.

## Tasks Completed
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

## New Realizations
### 1. J and Z Cannot Be Collected Like the Other Letters  
ASL letters **J** and **Z** involve **movement**, not static hand poses.  
This means:
- They cannot be captured using single-frame landmark snapshots.  
- They will require a different collection system that records sequences (video-based landmarks).  
- They should be excluded from the initial static dataset.

### 2. Dataset Progress (Completed Letters)
I have finished collecting dataset samples for the following ASL letters:

**A, B, C, D, E, F, G, H, I, K, L, M, N, S, T, U, V**

The goal is ~250 samples per letter, and currently these letters have completed sets.

## Notes & Challenges
- Some samples were unusable due to:
  - The hand moving too fast,
  - The camera missing landmarks,
  - Double-detected or partially detected hands.
- These bad samples will be removed or filtered out during the preprocessing stage.
- Collecting too many samples at once makes the hand tired — working in shorter sessions produces better results.
- Stable hand positioning improves MediaPipe detection accuracy significantly.