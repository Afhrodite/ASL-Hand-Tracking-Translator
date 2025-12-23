import cv2
import time
import numpy as np
from pathlib import Path

from hand_detector import HandDetector


# CONFIGURATION
VIDEO_DIR = Path("../data/raw_videos")
LANDMARK_DIR = Path("../data/video_landmarks")

FPS = 30  # Frames per second for video
DURATION_SEC = 120  # Max recording duration
VIDEO_NAME = "test_sentence"
SHOW_PREVIEW = True  # Show webcam window while recording


# HELPER FUNCTIONS
def get_unique_path(directory: Path, base_name: str, suffix):
    """
    Creates a file path that doesn't overwrite existing files.
    """
    path = directory / f"{base_name}{suffix}"
    counter = 1
    while path.exists():
        path = directory / f"{base_name}_{counter}{suffix}"
        counter += 1
    return path


# MAIN PIPELINE
def record_video_with_landmarks():
    # Make sure output directories exist
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    LANDMARK_DIR.mkdir(parents=True, exist_ok=True)

    # Create unique paths for video and landmark file
    video_path = get_unique_path(VIDEO_DIR, VIDEO_NAME, ".mp4")
    landmark_path = LANDMARK_DIR / (video_path.stem + ".npy")

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("❌ Could not open webcam")

    # Get webcam resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer (mp4)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, FPS, (width, height))

    # Initialize hand detector
    detector = HandDetector(
        mode=False,  # Continuous video mode
        maxHands=1,  # Only one hand
        detectionConfidence=0.5,
        trackConfidence=0.5,
    )

    # Store landmarks for every frame
    all_landmarks = []

    print(f"Recording → {video_path}")
    print(f"Saving landmarks → {landmark_path}")

    start_time = time.time()

    # Main recording loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect and draw hand landmarks on the frame
        detector.findHands(frame, draw=True)

        # Get landmark positions
        lm_list = detector.findPosition(frame, draw=False)

        # Initialize empty landmark frame (21 landmarks × x,y,z)
        frame_landmarks = np.zeros((21, 3), dtype=np.float32)

        # Fill landmark array if a hand is detected
        if lm_list:
            for lm in lm_list:
                idx, x, y, z = lm
                frame_landmarks[idx] = [x, y, z]

        # Save landmarks for this frame
        all_landmarks.append(frame_landmarks)

        # Save video frame
        out.write(frame)

        # Show preview window
        if SHOW_PREVIEW:
            cv2.imshow("Recording (press Q to stop)", frame)

        # Stop after duration or if user presses Q
        if time.time() - start_time >= DURATION_SEC:
            break

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Convert landmark list to numpy array and save
    landmarks_array = np.array(all_landmarks)
    np.save(landmark_path, landmarks_array)

    print("Recording finished")
    print(f"Landmarks shape: {landmarks_array.shape}")


if __name__ == "__main__":
    record_video_with_landmarks()
