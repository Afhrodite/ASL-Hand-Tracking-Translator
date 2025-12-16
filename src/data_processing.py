import os
import numpy as np
from sklearn.model_selection import train_test_split


# CONFIGURATION
# Folders
RAW_STATIC_DIR = "../data/raw_landmarks"
RAW_MOVEMENT_DIR = "../data/movement_sequences"
PROCESSED_DIR = "../data/processed"

# Static ASL letters
STATIC_LETTERS = [
    "A","B","C","D","E","F","G","H","I",
    "K","L","M","N","S","T","U","V","W","X","Y"
]

# Movement ASL letters
MOVEMENT_LETTERS = ["J", "Z"]

# Sequence lengths 
SEQUENCE_LENGTH_J = 40
SEQUENCE_LENGTH_Z = 60

# Hand landmark info
NUM_LANDMARKS = 21
COORDS_PER_LANDMARK = 3
FEATURES = NUM_LANDMARKS * COORDS_PER_LANDMARK

os.makedirs(PROCESSED_DIR, exist_ok=True)


# HELPER FUNCTIONS
def normalize_landmarks(landmarks):
    """
    Normalize landmarks by:
    - Using the wrist as the origin
    - Scaling by the max distance from the wrist

    This makes the data more consistent
    across different hand sizes and positions.
    """
    landmarks = landmarks.reshape(NUM_LANDMARKS, 3)

    # Wrist is landmark 0
    wrist = landmarks[0]
    landmarks = landmarks - wrist

    # Scale by maximum distance
    max_dist = np.max(np.linalg.norm(landmarks, axis=1))
    if max_dist > 0:
        landmarks = landmarks / max_dist

    return landmarks.flatten()


# STATIC DATA LOADING
def load_static_data():
    """
    Load and normalize static ASL letter data.
    Each sample is a single frame.
    """
    X, y = [], []

    for label, letter in enumerate(STATIC_LETTERS):
        letter_dir = os.path.join(RAW_STATIC_DIR, letter)
        if not os.path.exists(letter_dir):
            continue

        for file in os.listdir(letter_dir):
            if not file.endswith(".npy"):
                continue

            data = np.load(os.path.join(letter_dir, file))

            # Skip invalid samples (invalid shape)
            if data.shape != (FEATURES,):
                continue

            data = normalize_landmarks(data)

            X.append(data)
            y.append(label)

    return np.array(X), np.array(y)


# MOVEMENT DATA LOADING
def load_movement_letter(letter):
    """
    Load movement sequences for a single letter (J or Z).
    J and Z are handled separately.
    """
    X, y = [], []

    letter_dir = os.path.join(RAW_MOVEMENT_DIR, letter)
    if not os.path.exists(letter_dir):
        return np.array(X), np.array(y)

    # Pick correct sequence length
    seq_len = SEQUENCE_LENGTH_J if letter == "J" else SEQUENCE_LENGTH_Z
    label = MOVEMENT_LETTERS.index(letter)

    for file in os.listdir(letter_dir):
        if not file.endswith(".npy"):
            continue

        sequence = np.load(os.path.join(letter_dir, file))

        # Skip invalid sequences
        if sequence.shape != (seq_len, FEATURES):
            continue

        # Normalize each frame
        norm_sequence = []
        for frame in sequence:
            norm_sequence.append(normalize_landmarks(frame))

        X.append(norm_sequence)
        y.append(label)

    return np.array(X), np.array(y)


# SPLIT AND SAVE
def split_and_save(X, y, name, stratify_labels=True):
    """
    Split data into train / validation / test
    and save them as .npy files.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42,
        stratify=y if stratify_labels else None
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42,
        stratify=y_temp if stratify_labels else None
    )

    np.save(os.path.join(PROCESSED_DIR, f"{name}_X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DIR, f"{name}_y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, f"{name}_X_val.npy"), X_val)
    np.save(os.path.join(PROCESSED_DIR, f"{name}_y_val.npy"), y_val)
    np.save(os.path.join(PROCESSED_DIR, f"{name}_X_test.npy"), X_test)
    np.save(os.path.join(PROCESSED_DIR, f"{name}_y_test.npy"), y_test)


# MAIN PIPELINE
def main():
    print("Processing static ASL letters...")
    X_static, y_static = load_static_data()
    split_and_save(X_static, y_static, "static", stratify_labels=True)

    print("Processing movement letter J...")
    X_J, y_J = load_movement_letter("J")
    split_and_save(X_J, y_J, "movement_J", stratify_labels=False)

    print("Processing movement letter Z...")
    X_Z, y_Z = load_movement_letter("Z")
    split_and_save(X_Z, y_Z, "movement_Z", stratify_labels=False)

    print("Data processing complete!")


if __name__ == "__main__":
    main()  
            
    
    