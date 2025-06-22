from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
import pygame
import threading
import time
from queue import Queue
import numpy as np
import os

app = Flask(__name__)

# Global variables
camera = None
output_frame = None
fire_detected = False
is_running = False
frame_lock = threading.Lock()
model = None  # Global model variable

def init_camera():
    global camera
    try:
        if camera is not None:
            camera.release()
            time.sleep(0.5)  # Give some time for the camera to properly release
        
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow on Windows
        if not camera.isOpened():
            print("Error: Could not open camera")
            return False
            
        # Set camera properties for better performance
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size
        
        # Read a test frame to ensure camera is working
        ret, _ = camera.read()
        if not ret:
            print("Error: Could not read from camera")
            camera.release()
            return False
            
        return True
    except Exception as e:
        print(f"Error initializing camera: {str(e)}")
        if camera is not None:
            camera.release()
        return False

def init_model():
    global model
    if model is None:  # Only initialize if not already initialized
        model_path = 'YOLOv8-Fire-and-Smoke-Detection/runs/detect/train/weights/best.pt'
        if not os.path.exists(model_path):
            model_path = 'yolov8n.pt'
        model = YOLO(model_path)
        # Update class names to change 'default' to 'person'
        if 'default' in model.names:
            model.names[model.names.index('default')] = 'person'
    return model

def init_sound():
    pygame.mixer.init()
    sound_path = 'Fire-Truck-Sound.mp3'
    if os.path.exists(sound_path):
        return pygame.mixer.Sound(sound_path)
    return None

class DetectionThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.model = init_model()  # Use global model
        self.alarm_sound = init_sound()
        self.is_alarm_playing = False
        self.last_alarm_time = 0
        self.ALARM_COOLDOWN = 3
        self.running = True
        self.daemon = True  # Make thread daemon so it exits when main program exits

    def run(self):
        global output_frame, fire_detected, is_running
        
        while self.running and camera is not None and camera.isOpened():
            success, frame = camera.read()
            if not success:
                time.sleep(0.01)  # Reduced sleep time
                continue

            try:
                # Run detection with optimized settings
                results = self.model(frame, conf=0.25, iou=0.45)
                
                # Check for fire
                detected = False
                for r in results:
                    for c in r.boxes.cls:
                        class_name = self.model.names[int(c)]
                        if class_name.lower() == 'fire':
                            detected = True
                            break

                # Handle alarm with cooldown
                current_time = time.time()
                if detected and not self.is_alarm_playing and (current_time - self.last_alarm_time) > self.ALARM_COOLDOWN:
                    if self.alarm_sound:
                        self.alarm_sound.play()
                    self.is_alarm_playing = True
                    self.last_alarm_time = current_time
                elif not detected and self.is_alarm_playing:
                    if self.alarm_sound:
                        self.alarm_sound.stop()
                    self.is_alarm_playing = False

                # Update fire detection status
                fire_detected = detected
                
                # Draw detection results
                annotated_frame = results[0].plot()
                if detected:
                    cv2.putText(annotated_frame, 'FIRE DETECTED!', (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Update output frame
                with frame_lock:
                    output_frame = annotated_frame.copy()

            except Exception as e:
                print(f"Error in detection thread: {str(e)}")
                time.sleep(0.01)  # Reduced sleep time

    def stop(self):
        self.running = False
        if self.alarm_sound and self.is_alarm_playing:
            self.alarm_sound.stop()
        pygame.mixer.quit()

detection_thread = None

def generate_frames():
    global output_frame, camera
    while True:
        if output_frame is None or not is_running:
            time.sleep(0.01)
            continue

        try:
            with frame_lock:
                if output_frame is None:
                    continue
                frame = output_frame.copy()

            # Encode the frame with optimized parameters
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            ret, buffer = cv2.imencode('.jpg', frame, encode_param)
            if not ret:
                continue

            # Convert to bytes
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Error in generate_frames: {str(e)}")
            time.sleep(0.01)

@app.route('/')
def index():
    # Initialize model on startup to reduce initial delay
    init_model()
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify({'status': 'danger' if fire_detected else 'safe'})

@app.route('/start')
def start_detection():
    global detection_thread, is_running, camera
    if not is_running:
        # First try to initialize the camera
        if not init_camera():
            return jsonify({'status': 'error', 'message': 'Could not initialize camera'})
            
        try:
            detection_thread = DetectionThread()
            detection_thread.start()
            is_running = True
            return jsonify({'status': 'started'})
        except Exception as e:
            if camera is not None:
                camera.release()
            return jsonify({'status': 'error', 'message': str(e)})
    return jsonify({'status': 'already_running'})

@app.route('/stop')
def stop_detection():
    global detection_thread, is_running, camera, output_frame
    if is_running and detection_thread:
        try:
            detection_thread.stop()
            detection_thread.join(timeout=1.0)
            if camera is not None:
                camera.release()
                camera = None
            output_frame = None  # Clear the last frame
            is_running = False
            return jsonify({'status': 'stopped'})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)})
    return jsonify({'status': 'not_running'})

if __name__ == '__main__':
    # Ensure camera is released on startup
    if camera is not None:
        camera.release()
        camera = None
    output_frame = None
    
    # Initialize model on startup to reduce initial delay
    init_model()
    
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000) 