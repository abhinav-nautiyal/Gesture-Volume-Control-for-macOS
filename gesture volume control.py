import cv2
import time
import numpy as np
import handTrackingModule as htm
import math
import subprocess

################################
wCam, hCam = 640, 480
################################

# Open video capture
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

# Initialize hand detector
detector = htm.handDetector(detectionCon=0.7)

# Function to set volume on macOS using AppleScript
def set_volume(volume):
    # Volume is between 0 and 100 (0 = mute, 100 = max volume)
    volume = int(volume)
    script = f"osascript -e 'set volume output volume {volume}'"
    subprocess.run(script, shell=True)

# Set initial volume settings
vol = 0
volBar = 400
volPer = 0

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # Get thumb (4) and index (8) positions
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw landmarks and line between thumb and index finger
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        # Calculate the distance between thumb and index finger
        length = math.hypot(x2 - x1, y2 - y1)

        # Map hand range (50-300) to volume range (0-100)
        vol = np.interp(length, [50, 300], [0, 100])
        volBar = np.interp(length, [50, 300], [400, 150])
        volPer = np.interp(length, [50, 300], [0, 100])

        # Set the system volume using AppleScript (0-100)
        set_volume(vol)

        if length < 50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

    # Draw volume bar and percentage
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)

    # Calculate FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)

    # Show the video feed
    cv2.imshow("Img", img)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on pressing 'q'
        break