import cv2
import pyttsx3
import speech_recognition as sr
from ultralytics import YOLO
from collections import deque
import threading

model = YOLO("yolo11n.pt") 

cap = cv2.VideoCapture(0)

engine = pyttsx3.init()

engine.setProperty('rate', 170) 
engine.setProperty('volume', 1)  

memory_buffer = {}
next_object_id = 0  
memory_duration = 5

real_object_width = 0.5  


focal_length = 800 

is_known_environment = True 

def speak(text):
    engine.say(text)
    engine.runAndWait()

def generate_object_id():
    """Generate a unique ID for each object"""
    global next_object_id
    object_id = next_object_id
    next_object_id += 1  # Increment the object ID for the next detection
    return object_id

def calculate_distance(real_object_width, focal_length, object_width_in_frame):
    """Calculate the distance to the object based on its width in the frame"""
    if object_width_in_frame == 0:
        return 0
    return (real_object_width * focal_length) / object_width_in_frame

# Initialize the recognizer
recognizer = sr.Recognizer()

def listen_for_commands():
    """Listen for the 'change' command to toggle the environment"""
    global is_known_environment
    while True:
        with sr.Microphone() as source:
            print("Listening for 'change' command...")
            recognizer.adjust_for_ambient_noise(source)
            try:
                audio = recognizer.listen(source, timeout=5)  # Add timeout to avoid indefinite waiting
                command = recognizer.recognize_google(audio).lower()
                print(f"Command received: {command}")

                # If 'change' is detected, toggle the environment
                if "change" in command:
                    is_known_environment = not is_known_environment
                    environment_status = "known" if is_known_environment else "unknown"
                    speak(f"Environment switched to {environment_status}.")  # Speak the environment toggle
            except sr.UnknownValueError:
                # In case no command is recognized, continue listening
                continue
            except sr.RequestError:
                # Handle issues with the API request
                print("Could not request results from Google Speech Recognition service.")
                break

# Start the command listener in a separate thread to avoid blocking the main program
def start_command_listener():
    listen_for_commands()  # Start the infinite loop for listening



while True:
    listener_thread = threading.Thread(target=start_command_listener, daemon=True)
    listener_thread.start()
    # Capture frame-by-frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Use the YOLO model to make predictions on the frame
    results = model.track(frame)  # Use tracker to track objects in real time

    # Extract bounding boxes and other details
    boxes = results[0].boxes  # Contains bounding boxes, classes, and confidences

    # Loop through all detected boxes
    for box in boxes:
        # Extract the class index, confidence, and bounding box coordinates
        class_idx = int(box.cls)  # Class index
        confidence = box.conf.item()  # Confidence score
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # Bounding box coordinates
        class_name = results[0].names[class_idx]  # Get the class name

        # Initialize object_id for each detected object
        object_id = None

        # Assign a unique object ID
        for obj_id, coords in memory_buffer.items():
            if (x1, y1, x2, y2) in coords:
                object_id = obj_id
                break

        # If no existing ID is found, generate a new object ID
        if object_id is None:
            object_id = generate_object_id()

        # Update the memory buffer for tracking
        if object_id not in memory_buffer:
            memory_buffer[object_id] = deque(maxlen=memory_duration)
        memory_buffer[object_id].append((x1, y1, x2, y2))  # Store the object’s coordinates

        # Calculate the width of the object in the frame
        object_width_in_frame = x2 - x1

        # Calculate the distance to the object
        distance = calculate_distance(real_object_width, focal_length, object_width_in_frame)

        # Modify the speech output conditions based on the environment type
        if confidence > 0.5 and distance < 2:  # Only speak if the confidence is above a threshold
            if is_known_environment:
                speak(f"Known Environemnt: {class_name} with confidence {confidence:.2f} ")
                if distance < 1:
                    speak(f"Warning! {class_name} is {distance:.2f} meters away.")
            else:
                speak(f"Unknown Environment: {class_name} with confidence {confidence:.2f}")
                if distance < 1.5:
                    speak(f"Warning! The unknown object is {distance:.2f} meters away.")

        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{class_name} ID:{object_id} {confidence:.2f} Dist: {distance:.2f}m", 
                    (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Real-time Object Detection and Distance', frame)

    # Quit the loop when 'q' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit the loop when 'q' is pressed
        break

# Release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
