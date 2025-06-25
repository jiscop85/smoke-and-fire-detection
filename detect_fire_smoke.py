from ultralytics import YOLO
import cv2
import numpy as np
import pygame
import time

# Initialize pygame mixer for playing sound
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound('Fire-Truck-Sound.mp3')
is_alarm_playing = False
last_alarm_time = 0
ALARM_COOLDOWN = 3  # seconds between alarm triggers

# Load the model
model = YOLO('YOLOv8-Fire-and-Smoke-Detection/runs/detect/train/weights/best.pt')  # load the trained fire and smoke model

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam

# Set window name
window_name = "Fire and Smoke Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

while cap.isOpened():
    # Read a frame from webcam
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)
        
        # Check if fire is detected
        fire_detected = False
        for r in results:
            for c in r.boxes.cls:
                class_name = model.names[int(c)]
                if class_name.lower() == 'fire':
                    fire_detected = True
                    break
        
        # Handle alarm with cooldown
        current_time = time.time()
        if fire_detected and not is_alarm_playing and (current_time - last_alarm_time) > ALARM_COOLDOWN:
            alarm_sound.play()
            is_alarm_playing = True
            last_alarm_time = current_time
        elif not fire_detected and is_alarm_playing:
            alarm_sound.stop()
            is_alarm_playing = False
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        # Add warning text if fire is detected
        if fire_detected:
            cv2.putText(annotated_frame, 'FIRE DETECTED!', (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display the annotated frame
        cv2.imshow(window_name, annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release the video capture object and close the display window
if is_alarm_playing:
    alarm_sound.stop()
pygame.mixer.quit()
cap.release()
cv2.destroyAllWindows() 