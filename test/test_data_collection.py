import os
import numpy as np
import matplotlib.pyplot as plt

# CONFIGURATION
LETTER = "A"  # Change to the letter you want to check

# BASE_DIR
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# DATA_DIR
DATA_DIR = os.path.join(BASE_DIR, "test_data", LETTER)

EXPECTED_LANDMARKS = 21
COORDS_PER_LANDMARK = 3
EXPECTED_SHAPE = EXPECTED_LANDMARKS * COORDS_PER_LANDMARK

# Make sure the folder exists
if not os.path.exists(DATA_DIR):
    print(f"Folder {DATA_DIR} does not exist. Please create it and add .npy files.")
    exit()

# List all .npy files
files = [f for f in os.listdir(DATA_DIR) if f.endswith(".npy")]
print(f"Found {len(files)} .npy files in {DATA_DIR}\n")

good_samples = []
bad_samples = []

for f in files:
    path = os.path.join(DATA_DIR, f)
    arr = np.load(path)
    
    if arr.shape[0] != EXPECTED_SHAPE:
        print(f"[BAD] {f}: Unexpected shape {arr.shape}")
        bad_samples.append(f)
        continue

    # If shape is okay, we consider it "good" for now
    good_samples.append(f)

    # Visual check
    landmarks = arr.reshape(EXPECTED_LANDMARKS, COORDS_PER_LANDMARK)
    x = landmarks[:, 0]
    y = landmarks[:, 1]

    plt.figure(figsize=(4, 4))
    plt.scatter(x, y, c='r')
    plt.title(f"{f}")
    plt.gca().invert_yaxis()  # Flip y-axis because image origin is top-left
    plt.show()

print("\nSummary:")
print(f"Good samples: {len(good_samples)}")
print(f"Bad samples: {len(bad_samples)}")

if bad_samples:
    print("Bad samples:")
    for b in bad_samples:
        print(" -", b)