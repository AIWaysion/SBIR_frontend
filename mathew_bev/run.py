import os
import cv2
import numpy as np
import csv
import ultralytics
from ultralytics import YOLO
import time
from flask import Flask, Response
import threading

app = Flask(__name__)

# Global variables
global_frame = None
frame_lock = threading.Lock()

# Detection Model
model_path = "best.pt" # SBIR Dataset
model_path_East = "beste.pt" # SBIR Bellevue_7E Dataset

model = YOLO(model_path)
model_East = YOLO(model_path_East)

# Video Inputs
video_paths = [
    "rtsp://wowza01.bellevuewa.gov:1935/live/CCTV047.stream",
    "rtsp://wowza01.bellevuewa.gov:1935/live/CCTV047W.stream",
    "rtsp://wowza01.bellevuewa.gov:1935/live/CCTV047.stream", # For East View
    "rtsp://wowza01.bellevuewa.gov:1935/live/CCTV047S.stream",
    "rtsp://wowza01.bellevuewa.gov:1935/live/CCTV047.stream" # For Upper North View
]

# Output directories
output_directories = [
    "development/Comprehensive_Bellevue_BEV/Output_Directories/7N_Tracking_Data",
    "development/Comprehensive_Bellevue_BEV/Output_Directories/7W_Tracking_Data",
    "development/Comprehensive_Bellevue_BEV/Output_Directories/7N_Tracking_Data",
    "development/Comprehensive_Bellevue_BEV/Output_Directories/7S_Tracking_Data",
    "development/Comprehensive_Bellevue_BEV/Output_Directories/7N_Tracking_Data"
]

comprehensive_output_directory = "development/Comprehensive_Bellevue_BEV/Output_Directories/Comprehensive_Tracking_Data"

# Create output directories if they don't exist
for output_directory in output_directories:
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

csv_files = [
    os.path.join(output_directories[0], "development/Comprehensive_Bellevue_BEV/Output_Directories/7N_Tracking_Data/converted_track_history.csv"),
    os.path.join(output_directories[1], "development/Comprehensive_Bellevue_BEV/Output_Directories/7W_Tracking_Data/converted_track_history.csv"),
    os.path.join(output_directories[2], "development/Comprehensive_Bellevue_BEV/Output_Directories/7S_Tracking_Data/converted_track_history.csv"),
    os.path.join(output_directories[3], "development/Comprehensive_Bellevue_BEV/Output_Directories/7S_Tracking_Data/converted_track_history.csv"),
    os.path.join(output_directories[4], "development/Comprehensive_Bellevue_BEV/Output_Directories/7N_Upper_Tracking_Data/converted_track_history.csv")
]

# Background Image Path
background_image_path = "Major_View_Virtual_Background_SBIR.png"
background_image = cv2.imread(background_image_path)

comprehensive_output_directory_2 = "development/SBIR_CDA_Clips/SBIR_Major_View_MP4_Clip"

# Video Writer Parameters
output_video_path = os.path.join(comprehensive_output_directory_2, "comprehensive_output.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
fps = 13  # Adjust as per input video FPS
frame_width = background_image.shape[1]  # Full background image width
frame_height = background_image.shape[0]  # Full background image height

# Initialize VideoWriter
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Class-Color Mapping and other configurations from your original code
vehicle_colors = {
    "person": (255, 0, 0),      # Blue
    "class1": (0, 255, 0),      # Green
    "class2": (0, 0, 255),      # Red
    "class3": (0, 165, 255),    # Orange
    "class4": (255, 0, 255),    # Magenta
    "class5": (255, 255, 0),    # Cyan
    "class6": (128, 128, 128),  # Gray
    "class7": (0, 0, 128),      # Maroon
    "class8": (0, 128, 0),      # Dark Green
    "class9": (128, 0, 0),      # Navy
    "class10": (0, 128, 128),   # Olive
    "class11": (128, 0, 128),   # Purple
    "class12": (128, 128, 0),   # Teal
    "class13": (0, 165, 255),   # Orange
}

# According to .yaml file
class_id_to_name = {
    0: "person",
    3: "class1",
    2: "class2",
    5: "class4",
    80: "class3",
    81: "class5",
    82: "class6",
    83: "class7",
    84: "class8",
    85: "class9",
    86: "class10",
    87: "class11",
    88: "class12",
    89: "class13"
}

# Dilemma Zone & Ped Zone Definitions
dilemma_zones = [
    {"type": "polygon", "points": [(1035.94, 820.62), (1035.94, 841.17), (1754.93, 838.60), (1754.93, 824.48)]},
    {"type": "polygon", "points": [(1037.22, 843.74), (1037.20, 859.10), (1754.93, 854.01), (1754.93, 841.17)]},
    {"type": "polygon", "points": [(1038.51, 861.71), (1038.51, 877.12), (1754.93, 873.27), (1754.93, 857.86)]},
    {"type": "polygon", "points": [(888.29, 996.52), (906.26, 996.52), (907.55, 1476.71), (887.00, 1476.71)]},
    {"type": "polygon", "points": [(910.11, 996.52), (926.80, 996.50), (924.24, 1476.71), (908.80, 1476.70)]},
    {"type": "polygon", "points": [(929.40, 997.80), (944.80, 997.80), (939.70, 1478.00), (925.52, 1477.99)]},
    {"type": "polygon", "points": [(749.62, 861.71), (749.62, 878.40), (7.50, 870.70), (7.50, 852.70)]},
    {"type": "polygon", "points": [(7.50, 872.00), (749.62, 879.69), (749.62, 897.66), (7.50, 886.10)]},
    {"type": "polygon", "points": [(7.50, 887.40), (7.50, 904.10), (749.62, 916.92), (749.62, 898.94)]},
    {"type": "polygon", "points": [(906.30, 738.40), (888.30, 738.50), (887.00, 93.93), (906.30, 93.90)]},
    {"type": "polygon", "points": [(887.00, 738.45), (870.31, 738.45), (871.60, 7.90), (885.70, 7.90)]},
    {"type": "polygon", "points": [(852.30, 738.50), (869.00, 738.50), (870.31, 7.90), (854.90, 7.90)]},
    {"type": "polygon", "points": [(851.10, 738.50), (853.60, 7.90), (838.20, 7.90), (836.90, 738.50)]},
    {"type": "polygon_ped", "points": [(952.48, 806.85), (965.85, 815.77), (971.05, 920.53), (956.94, 930.93)]},
    {"type": "polygon_ped", "points": [(833.38, 778.47), (940.37, 780.70), (949.30, 797.00), (830.40, 793.33)]},
    {"type": "polygon_ped", "points": [(796.22, 821.56), (811.08, 813.39), (814.06, 928.55), (799.94, 918.89)]},
    {"type": "polygon_ped", "points": [(826.69, 942.67), (947.80, 935.20), (939.60, 950.10), (833.40, 956.80)]}
]

# Camera-Specific Parameters
camera_params = [
    { # Bellevue_7N_1
        'video_path': video_paths[0],
        'output_directory': output_directories[0],
        'csv_file': csv_files[0],

        'entry_zones': {
            "entry_zone_0": {
                "points": [
                    [137.31, 44.57],
                    [174.91, 45.10],
                    [173.80, 109.00],
                    [137.30, 109.00]
                ]
            },
            "entry_zone_1": {
                "points": [
                    [39.57, 48.86],
                    [69.65, 48.86],
                    [69.65, 105.79],
                    [39.57, 106.33]
                ]
            },
            "entry_zone_2": {
                "points": [
                    [70.70, 48.90],
                    [100.80, 48.80],
                    [101.30, 106.80],
                    [70.70, 106.40]
                ]
            },
            "entry_zone_3": {
                "points": [
                    [102.40, 107.40],
                    [135.70, 107.90],
                    [135.70, 48.90],
                    [101.90, 48.30]
                ]
            },
            "entry_zone_4": {
                "points": [
                    [291.44, 191.18],
                    [389.70, 191.20],
                    [390.30, 229.80],
                    [292.52, 230.38]
                ]
            },
            "entry_zone_5": {
                "points": [
                    [293.05, 232.53],
                    [396.70, 232.53],
                    [397.78, 262.60],
                    [293.60, 263.10]
                ]
            },
            "entry_zone_6": {
                "points": [
                    [294.13, 264.21],
                    [398.31, 264.21],
                    [398.85, 297.51],
                    [294.70, 298.00]
                ]
            }
        },
        'geo_coords': {
            'top_left': (47.627974, -122.144081),
            'top_right': (47.627965, -122.143321),
            'bottom_left': (47.627775, -122.144085),
            'bottom_right': (47.627762, -122.143310)
        },
        'pixel_points': {
            'top_left': (0, 0),
            'top_right': (461.3 - 1, 0),
            'bottom_left': (0, 377.9 - 1),
            'bottom_right': (461.3 - 1, 377.9 - 1)
        },
        'SOURCE': np.array([[475.5, 267.5], [908.5, 251.2], [1209.9, 385.5], [434.6, 577.5]]),
        'TARGET_WIDTH': 46.13,
        'TARGET_HEIGHT': 37.79,
        'scale_factor': 10,
        'CROP_X1': 814,
        'CROP_Y1': 721,
        'CROP_X2': 1056,
        'CROP_Y2': 921
    },
    { # Bellevue_7W
        'video_path': video_paths[1],
        'output_directory': output_directories[1],
        'csv_file': csv_files[1],
        'entry_zones': {
            "entry_zone_0": {
                "points": [
                    [470.40,98.30],
                    [550.18,99.87],
                    [551.20,133.30],
                    [469.50,130.80]
                ]
            },
            "entry_zone_1": {
                "points": [
                    [485.70,132.80],
                    [550.70,134.90],
                    [552.21,167.30],
                    [485.20,166.20]
                ]
            },
            "entry_zone_2": {
                "points": [
                    [485.70,167.80],
                    [552.21,168.82],
                    [553.20,199.20],
                    [485.60,196.70]
                ]
            }
        },
        'geo_coords': {
            'top_left': (47.627974, -122.144081),
            'top_right': (47.627965, -122.143321),
            'bottom_left': (47.627775, -122.144085),
            'bottom_right': (47.627762, -122.143310)
        },
        'pixel_points': {
            'top_left': (0, 0),
            'top_right': (583 - 1, 0),
            'bottom_left': (0, 219 - 1),
            'bottom_right': (583 - 1, 219 - 1)
        },
        'SOURCE': np.array([[528.5, 109.1], [879.3, 255.7], [683.5, 508.1], [324.6, 117.3]]),
        'TARGET_WIDTH': 58.37,
        'TARGET_HEIGHT': 21.98,
        'scale_factor': 10,
        'CROP_X1': 508,
        'CROP_Y1': 809,
        'CROP_X2': 811,
        'CROP_Y2': 928
    },
    { # Bellevue_7E
        # Parameters for camera 3
        'video_path': video_paths[2],
        'output_directory': output_directories[2],
        'csv_file': csv_files[2],
        'entry_zones':  {
            # "entry_zone_0": {"points": [[75.50, 57.00], [109.80, 55.50], [109.00, 128.90], [74.10, 128.90]]},
            # "entry_zone_1": {"points": [[112.60, 55.50], [137.50, 55.50], [137.50, 132.40], [111.20, 131.70]]},
            # "entry_zone_2": {"points": [[140.40, 55.50], [168.10, 56.20], [168.10, 132.50], [140.40, 132.40]]}
        },
        'geo_coords': {
            'top_left': (47.627974, -122.144081),
            'top_right': (47.627965, -122.143321),
            'bottom_left': (47.627775, -122.144085),
            'bottom_right': (47.627762, -122.143310)
        },
        'pixel_points': {
            'top_left': (0, 0),
            'top_right': (669 - 1, 0),
            'bottom_left': (0, 94 - 1),
            'bottom_right': (669 - 1, 94 - 1)
        },
        'SOURCE': np.array([[992.00,313.30], [1172.12,288.80], [1217.70,308.80], [1073.10,358.90]]),
        'TARGET_WIDTH': 66.86,
        'TARGET_HEIGHT': 9.35,
        'scale_factor': 10,
        'CROP_X1': 1035,
        'CROP_Y1': 825,
        'CROP_X2': 1385,
        'CROP_Y2': 876
    },
    { # Bellevue_7S
        # Parameters for camera 4
        'video_path': video_paths[3],
        'output_directory': output_directories[3],
        'csv_file': csv_files[3],
        'entry_zones':  {
            "entry_zone_0": {"points": [[75.50, 57.00], [109.80, 55.50], [109.00, 128.90], [74.10, 128.90]]},
            "entry_zone_1": {"points": [[112.60, 55.50], [137.50, 55.50], [137.50, 132.40], [111.20, 131.70]]},
            "entry_zone_2": {"points": [[140.40, 55.50], [168.10, 56.20], [168.10, 132.50], [140.40, 132.40]]}
        },
        'geo_coords': {
            'top_left': (47.627974, -122.144081),
            'top_right': (47.627965, -122.143321),
            'bottom_left': (47.627775, -122.144085),
            'bottom_right': (47.627762, -122.143310)
        },
        'pixel_points': {
            'top_left': (0, 0),
            'top_right': (223 - 1, 0),
            'bottom_left': (0, 500 - 1),
            'bottom_right': (223 - 1, 500 - 1)
        },
        'SOURCE': np.array([[352.9, 969.0], [99.7, 455.1], [720.6, 153.3], [982.0, 243.1]]),
        'TARGET_WIDTH': 22.32,
        'TARGET_HEIGHT': 50.00,
        'scale_factor': 10,
        'CROP_X1': 847,
        'CROP_Y1': 924,
        'CROP_X2': 971,
        'CROP_Y2': 1190
    },
    { # Bellevue_7N_Upper
        # Parameters for camera 4
        'video_path': video_paths[4],
        'output_directory': output_directories[4],
        'csv_file': csv_files[4],
        'entry_zones':  {
            # "entry_zone_0": {"points": [[75.50, 57.00], [109.80, 55.50], [109.00, 128.90], [74.10, 128.90]]},
            # "entry_zone_1": {"points": [[112.60, 55.50], [137.50, 55.50], [137.50, 132.40], [111.20, 131.70]]},
            # "entry_zone_2": {"points": [[140.40, 55.50], [168.10, 56.20], [168.10, 132.50], [140.40, 132.40]]}
        },
        'geo_coords': {
            'top_left': (47.627974, -122.144081),
            'top_right': (47.627965, -122.143321),
            'bottom_left': (47.627775, -122.144085),
            'bottom_right': (47.627762, -122.143310)
        },
        'pixel_points': {
            'top_left': (0, 0),
            'top_right': (224 - 1, 0),
            'bottom_left': (0, 297 - 1),
            'bottom_right': (224 - 1, 297 - 1)
        },
        'SOURCE': np.array([[523.70,215.40], [666.10,208.70], [734.00,253.20], [521.60,268.70]]),
        'TARGET_WIDTH': 22.43,
        'TARGET_HEIGHT': 29.67,
        'scale_factor': 10,
        'CROP_X1': 826,
        'CROP_Y1': 566,
        'CROP_X2': 949,
        'CROP_Y2': 720
    }
]

def compute_affine_transform(pixel_points, geo_points):
    A = []
    B = []
    for (px, py), (lat, lon) in zip(pixel_points, geo_points):
        A.append([px, py, 1, 0, 0, 0])
        A.append([0, 0, 0, px, py, 1])
        B.append(lat)
        B.append(lon)
    A = np.array(A)
    B = np.array(B)
    transform, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    return transform

def pixel_to_geo(pixel_coords, transform):
    x, y = pixel_coords
    lat = transform[0] * x + transform[1] * y + transform[2]
    lon = transform[3] * x + transform[4] * y + transform[5]
    return lat, lon

def process_video():
    global global_frame
    
    # Initialize Video Captures
    caps = [cv2.VideoCapture(params['video_path']) for params in camera_params]
    if not all([cap.isOpened() for cap in caps]):
        print("Error: Unable to open one or more video sources")
        return

    # Compute Transformations for Each Camera
    for params in camera_params:
        pixel_points = np.array([
            params['pixel_points']['top_left'],
            params['pixel_points']['top_right'],
            params['pixel_points']['bottom_left'],
            params['pixel_points']['bottom_right']
        ])
        geo_points = np.array([
            params['geo_coords']['top_left'],
            params['geo_coords']['top_right'],
            params['geo_coords']['bottom_left'],
            params['geo_coords']['bottom_right']
        ])
        params['transform'] = compute_affine_transform(pixel_points, geo_points)
        params['TARGET'] = np.array([
            [0, 0],
            [params['TARGET_WIDTH'] - 1, 0],
            [params['TARGET_WIDTH'] - 1, params['TARGET_HEIGHT'] - 1],
            [0, params['TARGET_HEIGHT'] - 1]
        ])
        params['m'] = cv2.getPerspectiveTransform(
            params['SOURCE'].astype(np.float32),
            params['TARGET'].astype(np.float32)
        )
        
        roi = background_image[params['CROP_Y1']:params['CROP_Y2'], 
                             params['CROP_X1']:params['CROP_X2']]
        top_down_width = int(params['TARGET_WIDTH'] * params['scale_factor'])
        top_down_height = int(params['TARGET_HEIGHT'] * params['scale_factor'])
        params['background_image_resized'] = cv2.resize(roi, 
                                                      (top_down_width, top_down_height))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    font_color = (0, 255, 0)

    frame_total_time = 0
    start_time_total = time.time()

    frame_count = 0
    while True:
        if all([cap.isOpened() for cap in caps]):
            frames = []
            for cap in caps:
                success, frame = cap.read()
                if not success:
                    break
                frames.append(frame)
            if len(frames) < len(caps):
                break

            frame_count += 1
            
            for idx, frame in enumerate(frames):
                params = camera_params[idx]
                results = model.predict(frame)
                    
                boxes = results[0].boxes.xywh.cpu()
                class_ids = results[0].boxes.cls.int().cpu().tolist()

                active_zones = {zone: False for zone in params['entry_zones']}
                top_down_view = params['background_image_resized'].copy()
                for zone, data in params['entry_zones'].items():
                    zone_points = np.array(data["points"], np.int32).reshape((-1, 1, 2))
                    cv2.polylines(top_down_view, [zone_points], isClosed=True, color=(255, 0, 0), thickness=1)

                # Process Detections
                for box, class_id in zip(boxes, class_ids):
                    x, y, w, h = box
                    
                    if idx == 0 or idx == 2 or idx == 4:  # 7N_1 & 7N_2 & 7E & 7N_Upper
                        y_point = y
                    else:
                        y_point = y + h / 2  # 7W & 7S
                        
                    transformed_points = cv2.perspectiveTransform(
                        np.array([[[x, y_point]]], dtype=np.float32),
                        params['m']
                    )
                    x_trans, y_trans = transformed_points[0, 0]
                    x_top_down = int(x_trans * params['scale_factor'])
                    y_top_down = int(y_trans * params['scale_factor'])
                    
                    # Map class_id to Class Name
                    vehicle_class = class_id_to_name.get(class_id, None)
                    if vehicle_class:
                        color = vehicle_colors.get(vehicle_class, (255, 255, 255))  # Default to white
                    else:
                        color = (255, 255, 255)
                    
                    cv2.circle(top_down_view, (x_top_down, y_top_down), 7, color, -1) # Non-Mean Version
                    
                
                    for zone, data in params['entry_zones'].items():
                        zone_polygon = np.array(data["points"], np.int32)
                        if cv2.pointPolygonTest(zone_polygon, (x_top_down, y_top_down), False) >= 0:
                            active_zones[zone] = True  # Mark zone as active
                            break

                    # Convert to lat/lon
                    lat, lon = pixel_to_geo((x_top_down, y_top_down), params['transform'])

                for i, (zone, data) in enumerate(params['entry_zones'].items()):
                    zone_points = np.array(data["points"], np.int32).reshape((-1, 1, 2))
                    color = (0, 0, 255) if active_zones[zone] else (255, 0, 0)  # Red if active, else blue
                    cv2.polylines(top_down_view, [zone_points], isClosed=True, color=color, thickness=2)

                    # Calculate center of the zone to place the zone number
                    zone_center = np.mean(data["points"], axis=0).astype(int)
                    cv2.putText(top_down_view, f"{i}", (zone_center[0], zone_center[1]), font, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
                
                # Resize the ROI back to the original size for full background image visualization with annotations
                roi_with_detections_resized_back = cv2.resize(top_down_view, (params['CROP_X2'] - params['CROP_X1'], params['CROP_Y2'] - params['CROP_Y1']))
                background_image[params['CROP_Y1']:params['CROP_Y2'], params['CROP_X1']:params['CROP_X2']] = roi_with_detections_resized_back
                
                for i, zone in enumerate(dilemma_zones):
                    if zone["type"] == "box":
                        xtl, ytl, xbr, ybr = int(zone["xtl"]), int(zone["ytl"]), int(zone["xbr"]), int(zone["ybr"])
                        cv2.rectangle(background_image, (xtl, ytl), (xbr, ybr), (0, 255, 255), 2)  # Yellow color
                    elif zone["type"] == "polygon":
                        points = np.array(zone["points"], np.int32).reshape((-1, 1, 2))
                        cv2.polylines(background_image, [points], isClosed=True, color=(0, 255, 255), thickness=2)
                    elif zone["type"] == "polygon_ped":
                        points = np.array(zone["points"], np.int32).reshape((-1, 1, 2))
                        cv2.polylines(background_image, [points], isClosed=True, color=(200, 170, 0), thickness=2)
            # Instead of cv2.imshow, update the global frame
            with frame_lock:
                global_frame = cv2.resize(background_image, 
                                        (int(background_image.shape[1] / 4),
                                        int(background_image.shape[0] / 4)))
        else:
            for cap in caps:
                cap.release()
            caps = [cv2.VideoCapture(params['video_path']) for params in camera_params]
    # Release resources
    for cap in caps:
        cap.release()

def generate_frames():
    while True:
        with frame_lock:
            if global_frame is not None:
                ret, buffer = cv2.imencode('.jpg', global_frame)
                if not ret:
                    continue
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)  # Approximately 30 FPS

@app.route('/video_feed')
def index():
    return """
    <html>
    <head>
        <title>Traffic Video Stream</title>
        <style>
            body { 
                margin: 0; 
                padding: 20px; 
                font-family: Arial, sans-serif;
                background-color: #f0f0f0;
            }
            h1 { 
                color: #333;
                text-align: center;
            }
            .video-container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            img {
                width: 100%;
                height: auto;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <h1>Traffic Monitoring System</h1>
        <div class="video-container">
            <img src="/video_feed">
        </div>
    </body>
    </html>
    """

@app.route('/')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start video processing in a separate thread
    video_thread = threading.Thread(target=process_video)
    video_thread.daemon = True
    video_thread.start()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, threaded=True)
