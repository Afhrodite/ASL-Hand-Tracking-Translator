import cv2
import os
import time
import numpy as np
from hand_detector import HandDetector


# CONFIGURATION
NUM_SAMPLES = 250  # How many samples per letter
PREP_TIME = 5  # Seconds before collection starts
DELAY_BETWEEN = 2  # Seconds between sample collecting


def main():
    # Ask which letter we collect
    letter = input("Enter the ASL letter you want to collect (A-Z): ").upper()

    save_dir = f"../data/raw_landmarks/{letter}"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nPreparing to collect samples for letter '{letter}'")
    print(f"Samples to collect: {NUM_SAMPLES}")
    print(f"Prep time: {PREP_TIME} seconds\n")

    # Initialize webcam + hand detector
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    # Prep countdown
    for sec in range(PREP_TIME, 0, -1):
        print(f"Starting in {sec} seconds...", end="\r")
        time.sleep(1)

    print("\nStarting data collection...\n")

    # Continue from existing sample count
    existing = len([f for f in os.listdir(save_dir) if f.endswith(".npy")])
    sample_count = existing

    print(f"Found {existing} existing samples. Starting from index {sample_count}.")

    target_total = existing + NUM_SAMPLES

    while sample_count < target_total:
        success, img = cap.read()
        if not success:
            continue

        # Detect hands
        img = detector.findHands(img, draw=True)
        lmList = detector.findPosition(img, draw=False)

        collected_now = sample_count - existing

        cv2.putText(
            img,
            f"Collecting letter: {letter}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            img,
            f"Sample: {collected_now}/{NUM_SAMPLES}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )

        cv2.imshow("Data Collection", img)
        cv2.waitKey(1)

        # Only save frame if a hand is detected
        if len(lmList) == 21:  # exactly 21 landmarks
            print(f"Saving sample {sample_count + 1}...")

            # Convert list-of-lists → flat numpy array
            coords = []
            for _, x, y, z in lmList:
                coords.extend([x, y, z])

            flat = np.array(coords)

            np.save(os.path.join(save_dir, f"{letter}_{sample_count}.npy"), flat)

            sample_count += 1

            # Wait so next sample isn't identical
            for sec in range(DELAY_BETWEEN, 0, -1):
                print(f"Next sample in {sec} seconds...", end="\r")
                time.sleep(1)

    print("\nDONE! Finished collecting samples.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
