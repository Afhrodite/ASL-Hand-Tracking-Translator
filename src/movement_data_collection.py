import cv2
import os
import time
import numpy as np
from hand_detector import HandDetector

# CONFIGURATION
SEQUENCE_LENGTH = 40  # Number of frames per movement sequence (~2 seconds)
NUM_SEQUENCES = 50  # How many sequences we want to collect
COUNTDOWN_TIME = 6  # Countdown seconds before each sequence starts


# Function to draw text in the center of the screen
def draw_center_text(img, text, scale=2, color=(0, 255, 255)):
    h, w, _ = img.shape
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 4)[0]
    x = (w - size[0]) // 2
    y = (h + size[1]) // 2
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 4)


def main():
    # Ask user which letter to collect (J or Z only)
    letter = input("Enter letter to collect (J or Z): ").upper()
    if letter not in ["J", "Z"]:
        return

    # Create folder to save sequences if it doesn't exist
    save_dir = f"../data/movement_sequences/{letter}"
    os.makedirs(save_dir, exist_ok=True)

    # Initialize webcam + hand detector
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)

    # Count how many sequences already exist
    existing = len([f for f in os.listdir(save_dir) if f.endswith(".npy")])
    seq_index = existing
    target_total = existing + NUM_SEQUENCES

    print(f"Collecting {NUM_SEQUENCES} samples for '{letter}'")

    while seq_index < target_total:

        # Countdown before recording
        for sec in range(COUNTDOWN_TIME, 0, -1):
            success, img = cap.read()
            if not success:  # Skip if webcam fails
                continue

            img = detector.findHands(img, draw=True)  # Detect hands
            draw_center_text(img, f"START IN {sec}")  # Show countdown on screen
            cv2.imshow("Movement Data Collection", img)
            cv2.waitKey(1)
            time.sleep(1)

        print(f"Recording sequence {seq_index}")

        frames_collected = 0  # Reset frame counter for this sequence
        sequence_data = []  # Store all frames in a list

        # Record frames for this sequence
        while frames_collected < SEQUENCE_LENGTH:
            success, img = cap.read()
            if not success:  # Skip if webcam fails
                continue

            # Detect hands
            img = detector.findHands(img, draw=True)
            lmList = detector.findPosition(img, draw=False)

            # Show "RECORDING" message and current frame
            cv2.putText(
                img,
                "RECORDING",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3,
            )

            cv2.putText(
                img,
                f"{frames_collected}/{SEQUENCE_LENGTH}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Movement Data Collection", img)
            cv2.waitKey(1)

            # Only save frame if exactly 21 landmarks detected
            if len(lmList) == 21:
                coords = []
                for _, x, y, z in lmList:  # Flatten landmark positions
                    coords.extend([x, y, z])

                sequence_data.append(coords)  # Add frame to sequence
                frames_collected += 1

        # Convert list to numpy array and save sequence
        np.save(
            os.path.join(save_dir, f"{letter}_{seq_index}.npy"), np.array(sequence_data)
        )

        print(f"Saved {letter}_{seq_index}")
        seq_index += 1  # Move to next sequence

    # Done collecting all sequences
    print("DONE")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
