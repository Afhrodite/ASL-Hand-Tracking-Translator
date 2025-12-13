# ASL Hand Tracking Translator
![Python](https://img.shields.io/badge/Python-3.x-blue)

... TO DO ...

_Created by **Réka Gábosi**_

## Table of Contents

... TO DO ...

## File Structure

```bash
asl-hand-tracking-translator/
│
├── data/                           # All collected dataset files
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
├── src/                            # Source code for the project
│   ├── hand_detector.py            # MediaPipe hand detection and landmark extraction
│   ├── data_collection.py          # Script for collecting static letter samples
│   └── movement_data_collection.py # Script for collecting J and Z movement sequences
│
├── images/                         # Images used for documentation
│   └── Hand_Landmarks.png          # Reference image showing hand landmark positions
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
