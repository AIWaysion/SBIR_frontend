from flask import Flask, Response, render_template
import cv2
import threading
import queue
from datetime import datetime
import numpy as np

app = Flask(__name__)

# store the processed frames
frame_queue = queue.Queue(maxsize=10)
stop_threads = False

def process_frame(frame):
    # todo: replace the logic here with tracking algorithm
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

def rtsp_stream():

    global stop_threads
    rtsp_url = "rtsp://wowza01.bellevuewa.gov:1935/live/CCTV047NE.stream"
    
    cap = cv2.VideoCapture(rtsp_url)
    
    while not stop_threads:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_frame(frame)
        
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        
        try:
            frame_queue.put_nowait(processed_frame)
        except queue.Full:
            pass
    
    cap.release()

def generate_frames():

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
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
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    rtsp_thread = threading.Thread(target=rtsp_stream)
    rtsp_thread.daemon = True
    rtsp_thread.start()
    
    app.run(host='0.0.0.0', port=5000)