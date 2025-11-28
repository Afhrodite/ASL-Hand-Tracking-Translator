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
