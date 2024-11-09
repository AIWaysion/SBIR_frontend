from flask import Flask, Response, render_template
import cv2
import threading
import queue
from datetime import datetime
import numpy as np

app = Flask(__name__)

# Store the processed frames for both streams
tracking_frame_queue = queue.Queue(maxsize=10)
visualization_frame_queue = queue.Queue(maxsize=10)
stop_threads = False

def process_tracking_frame(frame):
    # Process frame for tracking result
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # TODO: Add your tracking algorithm here
    return frame

def process_visualization_frame(frame):
    # Process frame for 3D visualization
    # TODO: Add your 3D visualization processing here
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, "3D View: " + timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return frame

def rtsp_stream():
    global stop_threads
    rtsp_url = "rtsp://wowza01.bellevuewa.gov:1935/live/CCTV047NE.stream"
    
    cap = cv2.VideoCapture(rtsp_url)
    
    while not stop_threads:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frames for both streams
        tracking_frame = process_tracking_frame(frame.copy())
        visualization_frame = process_visualization_frame(frame.copy())
        
        # Handle tracking frame queue
        if tracking_frame_queue.full():
            try:
                tracking_frame_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            tracking_frame_queue.put_nowait(tracking_frame)
        except queue.Full:
            pass
            
        # Handle visualization frame queue
        if visualization_frame_queue.full():
            try:
                visualization_frame_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            visualization_frame_queue.put_nowait(visualization_frame)
        except queue.Full:
            pass
    
    cap.release()

def generate_tracking_frames():
    while True:
        if not tracking_frame_queue.empty():
            frame = tracking_frame_queue.get()
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def generate_visualization_frames():
    while True:
        if not visualization_frame_queue.empty():
            frame = visualization_frame_queue.get()
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_tracking_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_3d')
def video_feed_3d():
    return Response(generate_visualization_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    rtsp_thread = threading.Thread(target=rtsp_stream)
    rtsp_thread.daemon = True
    rtsp_thread.start()
    
    app.run(host='0.0.0.0', port=5000)