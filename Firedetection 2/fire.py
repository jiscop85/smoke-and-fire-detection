from ultralytics import YOLO
import cv2
import pygame
import os
import numpy as np
import time

def initialize_alarm():
    """Initialize the alarm system"""
    pygame.mixer.init()
    pygame.mixer.music.load('UpMusic - Sedaye azhire police (754).mp3')
    pygame.mixer.music.set_volume(1.0)  # Set maximum volume

def start_alarm():
    """Start playing the alarm sound"""
    if not pygame.mixer.get_busy():
        pygame.mixer.music.play(-1)  # -1 means loop indefinitely

def stop_alarm():
    """Stop the alarm sound"""
    if pygame.mixer.get_busy():
        pygame.mixer.music.stop()

def is_fire_color(hsv_image, x1, y1, x2, y2):
    """Check if the detected region has fire-like colors"""
    roi = hsv_image[y1:y2, x1:x2]
    
    # Define fire color ranges in HSV
    # Red range 1 (0-10)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    # Red range 2 (160-180)
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    # Orange/Yellow range
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    
    # Create masks for fire colors
    mask1 = cv2.inRange(roi, lower_red1, upper_red1)
    mask2 = cv2.inRange(roi, lower_red2, upper_red2)
    mask3 = cv2.inRange(roi, lower_yellow, upper_yellow)
    
    # Combine masks
    mask = cv2.bitwise_or(cv2.bitwise_or(mask1, mask2), mask3)
    
    # Calculate percentage of fire-like colors
    fire_percentage = (np.sum(mask) / 255) / (mask.shape[0] * mask.shape[1])
    
    # Check for color intensity and distribution
    if fire_percentage > 0.15:  # Lower threshold for initial detection
        # Calculate color intensity
        intensity = np.mean(roi[:, :, 2])  # V channel in HSV
        # Check for color distribution (fire should have varying intensities)
        std_intensity = np.std(roi[:, :, 2])
        
        return intensity > 100 and std_intensity > 30  # Fire should be bright and have varying intensities
    return False

def draw_fire_detection(frame, box, confidence):
    """Draw bounding box and label for fire detection"""
    x1, y1, x2, y2 = map(int, box)
    
    # Draw rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Prepare label text
    label_text = f'Fire: {confidence:.2f}'
    
    # Get text size for background rectangle
    (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    
    # Draw background rectangle for text
    cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 0, 255), -1)
    
    # Draw text
    cv2.putText(frame, label_text, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def load_model():
    """Load and optionally train the YOLO model"""
    try:
        # Try to load a pre-trained model specifically for fire detection
        model_path = 'fire_detection.pt'  # Custom fire detection model
        
        # Check if model exists, if not download it
        if not os.path.exists(model_path):
            print("Downloading fire detection model...")
            # Use a pre-trained model specifically for fire detection
            model = YOLO('yolov8n')  # Start with base model
            print("Training model for fire detection...")
            model.train(
                data='fire_data.yaml',
                epochs=50,
                imgsz=640,
                batch=16,
                patience=10,
                save=True,
                device='0' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
            )
            # Save the trained model
            model.save(model_path)
            print("Model trained and saved successfully!")
        else:
            model = YOLO(model_path)
            print("Loaded fire detection model successfully!")
            
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        # Fallback to base model if training fails
        try:
            print("Falling back to base model...")
            return YOLO('yolov8n')
        except Exception as e:
            print(f"Failed to load fallback model: {e}")
            return None

def main():
    # Initialize alarm system
    initialize_alarm()
    
    # Load YOLO model
    model = load_model()
    if model is None:
        print("Failed to load model. Exiting...")
        return

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("Press 'q' to quit")
    
    # Variables for detection stability
    detection_history = []
    history_size = 5  # Number of frames to consider for stable detection
    last_alarm_time = 0
    alarm_cooldown = 1.0  # Minimum time between alarm toggles (in seconds)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Convert frame to HSV for color detection
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Make prediction with higher confidence threshold
        results = model.predict(frame, conf=0.4)  # Lower confidence threshold for initial detection
        
        current_detection = False
        fire_detected = False

        # Process detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                # Only process fire detections
                if class_id == 0:  # Fire class
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Check both model and color detection
                    if is_fire_color(hsv_frame, x1, y1, x2, y2):
                        draw_fire_detection(frame, box.xyxy[0], confidence)
                        fire_detected = True
                        current_detection = True

        # Update detection history
        detection_history.append(current_detection)
        if len(detection_history) > history_size:
            detection_history.pop(0)

        # Check for stable detection
        stable_detection = sum(detection_history) >= (history_size * 0.6)  # 60% of recent frames

        # Control alarm based on stable detection
        current_time = time.time()
        if stable_detection and (current_time - last_alarm_time) >= alarm_cooldown:
            start_alarm()
            last_alarm_time = current_time
            # Add warning text
            cv2.putText(frame, "FIRE DETECTED!", (frame.shape[1]//2 - 150, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        elif not stable_detection:
            stop_alarm()

        # Add status text to frame
        status_text = "Status: "
        if fire_detected:
            status_text += "Fire Detected!"
        else:
            status_text += "No Fire Detected"
            
        cv2.putText(frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show frame
        cv2.imshow('Fire Detection', frame)

        # Break loop on 'q' press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    stop_alarm()

if __name__ == '__main__':
    main()