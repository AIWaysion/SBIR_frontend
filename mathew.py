import os
import cv2
import numpy as np
import csv
import ultralytics
from ultralytics import YOLO
import time
from flask import Flask, Response, render_template
import threading

app = Flask(__name__)

# Global variable to store the latest frame
global_frame = None
frame_lock = threading.Lock()

# BEV Entry Zone Definitions for Bellevue_7N_2
entry_zones = {
    "entry_zone_0": {
        "points": [
            [523.33, 441.90],
            [652.90, 440.70],
            [652.90, 470.80],
            [524.49, 470.82]
        ]
    },
    "entry_zone_1": {
        "points": [
            [650.57, 500.89],
            [650.57, 475.44],
            [525.64, 473.13],
            [525.64, 503.20]
        ]
    },
    "entry_zone_2": {
        "points": [
            [648.25, 536.75],
            [649.40, 505.50],
            [527.96, 505.52],
            [526.80, 536.75]
        ]
    },
    "entry_zone_3": {
        "points": [
            [405.35, 208.25],
            [370.65, 205.93],
            [369.49, 345.89],
            [404.19, 345.89]
        ]
    },
    "entry_zone_4": {
        "points": [
            [367.18, 207.09],
            [339.42, 205.93],
            [338.26, 342.42],
            [367.18, 345.89]
        ]
    },
    "entry_zone_5": {
        "points": [
            [335.95, 205.93],
            [308.19, 204.78],
            [307.03, 344.74],
            [335.95, 343.58]
        ]
    },
    "entry_zone_6": {
        "points": [
            [271.17, 205.93],
            [305.87, 204.78],
            [304.72, 344.74],
            [271.17, 343.58]
        ]
    },
    "entry_zone_7": {
        "points": [
            [204.10, 505.50],
            [67.60, 504.40],
            [65.28, 534.44],
            [205.24, 533.28]
        ]
    },
    "entry_zone_8": {
        "points": [
            [65.28, 537.91],
            [205.24, 536.75],
            [207.55, 566.82],
            [64.12, 571.45]
        ]
    },
    "entry_zone_9": {
        "points": [
            [206.40, 606.15],
            [205.24, 570.29],
            [65.28, 573.76],
            [66.44, 606.15]
        ]
    },
    "entry_zone_10": {
        "points": [
            [375.27, 679.02],
            [407.66, 677.87],
            [406.51, 794.69],
            [369.50, 795.80]
        ]
    },
    "entry_zone_11": {
        "points": [
            [412.29, 679.02],
            [452.80, 677.90],
            [451.60, 797.00],
            [409.98, 795.85]
        ]
    },
    "entry_zone_12": {
        "points": [
            [458.50, 677.80],
            [500.20, 679.00],
            [499.04, 794.69],
            [457.40, 795.80]
        ]
    }
}

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_thickness = 1
font_color = (0, 255, 0)
frame_count = 0

top_left_px = (0, 0)
top_right_px = (687 - 1, 0)
bottom_left_px = (0, 812 - 1)
bottom_right_px = (812 - 1, 687 - 1)

pixel_points = np.array([top_left_px, top_right_px, bottom_left_px, bottom_right_px])
# geo_points = np.array([top_left, top_right, bottom_left, bottom_right])

SOURCE = np.array([[12.20,367.20], [410.20,190.40], [1271.20,268.60], [-1300.20,6000.00]]) # Bellevue_7N_2 (New)
TARGET_HEIGHT = 81.21
TARGET_WIDTH = 68.74

TARGET = np.array([[0, 0], [TARGET_WIDTH - 1, 0], [TARGET_WIDTH - 1, TARGET_HEIGHT - 1], [0, TARGET_HEIGHT - 1]])
m = cv2.getPerspectiveTransform(SOURCE.astype(np.float32), TARGET.astype(np.float32))

def generate_frames():
    global global_frame
    while True:
        with frame_lock:
            if global_frame is not None:
                # Encode the frame as JPEG
                ret, buffer = cv2.imencode('.jpg', global_frame)
                if not ret:
                    continue
                # Convert to bytes and yield for MJPEG streaming
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.01)  # Small delay to prevent excessive CPU usage

@app.route('/')
def index():
    return render_template('mathew.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def process_video():
    global global_frame
    
    # Your existing configuration code here
    model_path = "weights/best_Near_Miss.pt"
    video_path = "rtsp://wowza01.bellevuewa.gov:1935/live/CCTV047NE.stream"
    background_image_path = "development/Bellevue_BEV_Background.png"
    model = YOLO(model_path)

    # Keep your existing entry_zones dictionary and other configurations...
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video source")
        return
    
    scale_factor = 10
    background_image = cv2.imread(background_image_path)
    CROP_X1, CROP_Y1 = 160, 13
    CROP_X2, CROP_Y2 = 908, 889
    
    roi = background_image[CROP_Y1:CROP_Y2, CROP_X1:CROP_X2]
    top_down_width = int(TARGET_WIDTH * scale_factor)
    top_down_height = int(TARGET_HEIGHT * scale_factor)
    background_image_resized = cv2.resize(roi, (top_down_width, top_down_height))

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model.predict(frame)
        top_down_view = background_image_resized.copy()
        boxes = results[0].boxes.xywh.cpu()
        
        active_zones = {zone: False for zone in entry_zones}
        
        # Your existing detection and drawing code...
        for idx, box in enumerate(boxes):
            x, y, w, h = box
            y_bottom = y + h / 2
            transformed_points = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), m)
            x_trans, y_trans = transformed_points[0, 0]
            x_top_down = int(x_trans * scale_factor)
            y_top_down = int(y_trans * scale_factor)
            
            cv2.circle(top_down_view, (x_top_down, y_top_down), 5, (0, 255, 0), -1)
            
            for zone, data in entry_zones.items():
                zone_polygon = np.array(data["points"], np.int32)
                if cv2.pointPolygonTest(zone_polygon, (x_top_down, y_top_down), False) >= 0:
                    active_zones[zone] = True
                    break

        # Draw entry zones
        for i, (zone, data) in enumerate(entry_zones.items()):
            zone_points = np.array(data["points"], np.int32).reshape((-1, 1, 2))
            color = (0, 0, 255) if active_zones[zone] else (255, 0, 0)
            cv2.polylines(top_down_view, [zone_points], isClosed=True, color=color, thickness=2)
            
            zone_center = np.mean(data["points"], axis=0).astype(int)
            cv2.putText(top_down_view, f"{i}", (zone_center[0], zone_center[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

        roi_with_detections_resized_back = cv2.resize(top_down_view, 
                                                     (CROP_X2 - CROP_X1, CROP_Y2 - CROP_Y1))
        background_image[CROP_Y1:CROP_Y2, CROP_X1:CROP_X2] = roi_with_detections_resized_back
        
        resized_display = cv2.resize(background_image, 
                                   (int(background_image.shape[0] / 4), 
                                    int(background_image.shape[1] / 4)))
        
        # Update the global frame with thread safety
        with frame_lock:
            global_frame = resized_display.copy()

    cap.release()

if __name__ == '__main__':
    # Start video processing in a separate thread
    video_thread = threading.Thread(target=process_video)
    video_thread.daemon = True
    video_thread.start()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)