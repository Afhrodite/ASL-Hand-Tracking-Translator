import cv2
import mediapipe as mp
import time


class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionConfidence=0.5, trackConfidence=0.5):
        """Store parameters for the hand detector"""
        self.mode = mode # If True, treats input as static images (not continuous video)
        self.maxHands = maxHands # Maximum number of hands to detect at once
        self.detectionConfidence = detectionConfidence # Minimum confidence for initial hand detection
        self.trackConfidence = trackConfidence # Confidence threshold for tracking detected hands

        # Initialize MediaPipe's Hands solution
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionConfidence,
            min_tracking_confidence=self.trackConfidence
        )

        # Utility class for drawing the hand landmarks on the image
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        # Convert the BGR image (OpenCV default) to RGB for MediaPipe processing
        imgRGB =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        # Run MediaPipe hand detection model on the frame
        self.results = self.hands.process(imgRGB)

        # If hands are detected, draw the landmarks on the original image
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # Draws the 21 hand landmarks + connections between them
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNumber=0, draw=True):
        # List to store all (id, x, y) for the detected hand landmarks
        lmList = []

        # if no hands → return empty list
        if not self.results or not self.results.multi_hand_landmarks:
            return lmList

        # Select specific hand
        myHand = self.results.multi_hand_landmarks[handNumber]


        # Loop through each of the 21 landmarks
        for id, lm in enumerate(myHand.landmark):
            # Convert normalized landmark coordinates to actual pixel coordinates
            h, w, c = img.shape
            cx, cy = int(lm.x*w), int(lm.y*h)

            # Append landmark id and its (x, y) position and z value
            lmList.append([id, cx, cy, lm.z])

            # Optionally draw a small circle on the selected landmarks
            if draw:
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return lmList
    
def main():
    # Initialize FPS timer
    pTime = 0 # Previous time
    cTime = 0 # Current time

    # Start webcam capture (0 = default camera)
    cap = cv2.VideoCapture(0)

    # Create an instance of our hand detector
    detector = HandDetector()

    # Main loop to continuously read frames from webcam
    while True:
        # Read a frame from the webcam
        success, img = cap.read()

        # Detect hands and draw landmarks
        img = detector.findHands(img)

        # Extract hand landmark positions
        lmList = detector.findPosition(img)

        # Print the coordinates of the first landmark if available
        if len(lmList) != 0:
            print(lmList[0]) # Prints: [landmark_id, x, y]

        # Calculate and display FPS (frames per second)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        # Show webcam image
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()