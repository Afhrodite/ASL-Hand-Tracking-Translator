import os
import numpy as np
import joblib
from collections import deque, Counter
from tensorflow.keras.models import load_model


# CONFIGURATION
LANDMARK_DIR = "../data/video_landmarks"
STATIC_MODEL_PATH = "../models/static_best_model_MLP.joblib"
J_MODEL_PATH = "../models/movement_J_best_LSTM.keras"
Z_MODEL_PATH = "../models/movement_Z_best_LSTM.keras"

# Alphabet mapping
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# How many frames are used to detect movement-based letters
MOVEMENT_WINDOWS = {"J": 40, "Z": 60}

# Minimum number of consecutive frames needed to accept a prediction
MIN_STABLE_FRAMES = 3

# Allow double letters only if gap is big enough
DOUBLE_LETTER_GAP = 12

# Number of frames used for majority voting
VOTE_WINDOW = 12


# LOAD MODELS
static_model = joblib.load(STATIC_MODEL_PATH)
movement_models = {
    "J": load_model(J_MODEL_PATH),
    "Z": load_model(Z_MODEL_PATH),
}


# HELPER FUNCTIONS
def flatten_landmarks(frame):
    """
    Normalizes and flattens 21 hand landmarks into a 1D vector.

    Steps:
    - Center landmarks relative to wrist
    - Scale to make size invariant
    - Flatten to feed into MLP
    """
    frame = frame.reshape(21, 3)

    # Use wrist as origin to remove global position
    wrist = frame[0]
    frame = frame - wrist

    # Normalize by maximum distance to make scale invariant
    max_dist = np.max(np.linalg.norm(frame, axis=1))
    if max_dist > 0:
        frame = frame / max_dist

    # Convert to shape (63,)
    return frame.flatten()


def predict_static(frame_84):
    """
    Predicts a static ASL letter from a single frame.
    """
    pred_label = static_model.predict([frame_84])[0]  # returns int
    return LETTERS[pred_label]


def predict_movement(letter, seq):
    """
    Predicts movement-based letters (J or Z) using an LSTM.
    """
    model = movement_models[letter]
    seq = np.array(seq, dtype=np.float32)

    timesteps = model.input_shape[1]

    # Ensure sequence matches model input length
    if len(seq) > timesteps:
        seq = seq[-timesteps:]
    elif len(seq) < timesteps:
        pad = np.zeros((timesteps - len(seq), seq.shape[1]), dtype=np.float32)
        seq = np.vstack((pad, seq))

    # Add batch dimension
    seq = np.expand_dims(seq, axis=0)

    # Predict probabilities
    probs = model.predict(seq, verbose=0)[0]  # shape: (num_classes,)
    pred_index = np.argmax(probs)

    # Convention: class 1 means movement detected
    if pred_index == 1:
        return letter
    return None


def remove_noise(preds, min_stable=MIN_STABLE_FRAMES):
    """
    Removes noisy predictions by keeping letters only if they
    appear consecutively for a minimum number of frames.
    """
    cleaned = []
    count = 0
    prev = None

    for p in preds:
        if p == prev:
            count += 1
        else:
            count = 1
            prev = p

        if count == min_stable:
            cleaned.append(p)

    return cleaned


def collapse_letters(preds, double_gap=DOUBLE_LETTER_GAP):
    """
    Collapses repeated letters unless they are far enough apart
    to be considered intentional (e.g. 'LL' in a word).
    """
    result = []
    last_pos = {}
    for i, p in enumerate(preds):
        if p not in last_pos or i - last_pos[p] > double_gap:
            result.append(p)
            last_pos[p] = i

    return "".join(result)


# MAIN PIPELINE
def run_on_video(npy_path):
    """
    Runs inference on a single landmark file and returns
    a string of predicted letters.
    """
    data = np.load(npy_path)

    vote_buffer = []  # Stores per-frame predictions
    final_output = []  # Stores voted predictions

    for frame in data:
        frame_84 = flatten_landmarks(frame)

        # Skip frames with no detected hand
        if np.sum(frame_84) == 0:
            continue

        # ONLY FOR STATIC MODEL
        letter = predict_static(frame_84)
        vote_buffer.append(letter)

        # Majority vote every VOTE_WINDOW frames
        if len(vote_buffer) == VOTE_WINDOW:
            voted = Counter(vote_buffer).most_common(1)[0][0]
            final_output.append(voted)
            vote_buffer.clear()

    # Handle leftover frames at the end
    if vote_buffer:
        voted = Counter(vote_buffer).most_common(1)[0][0]
        final_output.append(voted)

    return "".join(final_output)


# RUN ON ALL VIDEOS
if __name__ == "__main__":
    print("\n===== ASL PREDICTIONS =====")
    for file in os.listdir(LANDMARK_DIR):
        if not file.endswith(".npy"):
            continue

        path = os.path.join(LANDMARK_DIR, file)

        print("\n-------------------------------")
        print("Video:", file)
        prediction = run_on_video(path)
        print("Prediction:", prediction)
