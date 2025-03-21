import cv2
import numpy as np
import pyttsx3
import threading

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Function to speak a message
def speak(message):
    engine.say(message)
    engine.runAndWait()

# Initialize the camera (0 is the default camera)
cap = cv2.VideoCapture(0)

# Set the resolution of the camera (optional: lower the resolution to speed up processing)
cap.set(3, 320)  # width
cap.set(4, 240)  # height

# Function to check if an area has obstacles
def check_region(region_img, min_area=500):
    gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Count obstacles by area of contours
    obstacle_count = 0
    for contour in contours:
        if cv2.contourArea(contour) > min_area:  # Ignore small noise
            obstacle_count += 1
    return obstacle_count

# Function to handle real-time obstacle detection and feedback
def detect_and_respond(frame):
    # Split frame into regions: Left, Center, Right
    height, width, _ = frame.shape
    left_region = frame[:, :width//3]  # Left third of the frame
    center_region = frame[:, width//3:2*width//3]  # Center third of the frame
    right_region = frame[:, 2*width//3:]  # Right third of the frame

    # Check for obstacles in each region
    left_obstacles = check_region(left_region)
    center_obstacles = check_region(center_region)
    right_obstacles = check_region(right_region)

    # Provide directional feedback based on detected obstacles
    if left_obstacles == 0 and center_obstacles == 0 and right_obstacles == 0:
        speak("Path is clear. You can move forward.")
    elif left_obstacles == 0 and right_obstacles == 0:
        speak("Clear on both sides, move forward.")
    elif left_obstacles == 0:
        speak("Clear on the left, move left.")
    elif right_obstacles == 0:
        speak("Clear on the right, move right.")
    elif center_obstacles == 0:
        speak("Clear in front, move forward.")
    else:
        speak("Obstacle detected. Please backtrack or stop.")

    # Return the frame with regions for display
    return frame

# Main loop for video feed
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is captured successfully, process it
    if ret:
        # Run the obstacle detection
        frame = detect_and_respond(frame)

        # Show the frame with the regions
        cv2.imshow('Obstacle Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
