from shapely.geometry import Point, Polygon
import os
import sys
import argparse
import os

# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from random import random as ran
from datetime import datetime
from flask import Flask, Response, render_template
import threading
from queue import Queue
import pandas as pd
from collections import Counter

import warnings

warnings.filterwarnings("ignore")

import csv
from datetime import datetime

app = Flask(__name__)
frame_queue = Queue(maxsize=2)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # strongsort root directory
WEIGHTS = ROOT / "weights"


if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / "yolov7") not in sys.path:
    sys.path.append(str(ROOT / "yolov7"))  # add yolov7 ROOT to PATH
if str(ROOT / "strong_sort") not in sys.path:
    sys.path.append(str(ROOT / "strong_sort"))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadStreams, LoadImages
from yolov7.utils.general import (
    check_img_size,
    check_requirements,
    check_imshow,
    non_max_suppression,
    apply_classifier,
    scale_coords,
    xyxy2xywh,
    strip_optimizer,
    set_logging,
    increment_path,
)
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import (
    select_device,
    load_classifier,
    time_synchronized,
    TracedModel,
)

from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT

@app.route('/')
def index():
    return render_template('tracking.html')

def generate_frames():
    while True:
        try:
            frame = frame_queue.get()
            if frame is None:
                continue
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            # Construct MJPEG frame
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Error in generate_frames: {e}")
            continue

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def is_point_in_bbox(point, bbox):
    """
    Check if a point is inside a bounding box.

    Parameters:
    point (tuple): A tuple (x, y) representing the point coordinates.
    bbox (tuple): A tuple ((x1, y1), (x2, y2)) representing the bounding box,
                  where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.

    Returns:
    bool: True if the point is inside the bbox, False otherwise.
    """
    x, y = point
    (x1, y1), (x2, y2) = bbox

    return x1 <= x <= x2 and y1 <= y <= y2


# Define zones and their center points
# zones = {
#     "z11": [[250, 524], [224, 593], [312, 600], [335, 532]],
#     "z12": [[272, 473], [250, 524], [335, 532], [353, 480]],
#     "z13": [[289, 432], [272, 473], [353, 480], [369, 433]],
#     "z10": [[534, 249], [536, 299], [571, 296], [561, 247]],
#     "z9": [[561, 247], [571, 296], [620, 292], [594, 247]],
#     "z8": [[594, 247], [620, 292], [657, 286], [630, 245]],
#     "z7": [[630, 245], [657, 286], [690, 286], [655, 243]],
#     "z6": [[874, 315], [910, 338], [1003, 317], [976, 301]],
#     "z5": [[910, 338], [954, 361], [1030, 332], [1003, 317]],
#     "z4": [[954, 361], [984, 379], [1063, 354], [1030, 332]]
# }

WEST_ZONES = {
    "entry_zone_0": [
        [701.7, 272.2],
        [736.88, 251.08],
        [855.01, 316.14],
        [852.0, 316.1],
        [813.1, 340.9],
    ],
    "dilemma_zone_2": [[615.7, 320.3], [659.9, 292.7], [433.8, 141.1], [415.8, 146.1]],
    "dilemma_zone_3": [
        [430.0, 138.1],
        [449.6, 134.2],
        [695.4, 270.3],
        [660.62, 292.23],
    ],
    "dilemma_zone_4": [[416.9, 115.8], [430.5, 111.2], [733.4, 249.1], [697.2, 269.4]],
    "dilemma_zone_5": [
        [1084.44, 266.82],
        [1107.53, 281.14],
        [1125.3, 166.9],
        [1121.3, 165.3],
    ],
    "dilemma_zone_6": [
        [1110.05, 282.78],
        [1135.45, 299.64],
        [1131.6, 170.2],
        [1126.1, 167.4],
    ],
    "dilemma_zone_7": [
        [1138.6, 301.35],
        [1166.7, 319.44],
        [1138.6, 173.1],
        [1132.5, 170.4],
    ],
    "dilemma_zone_8": [
        [1139.4, 173.5],
        [1147.1, 177.1],
        [1198.0, 339.3],
        [1168.97, 320.87],
    ],
    "entry_zone_9": [
        [721.93, 411.34],
        [769.6, 374.7],
        [661.4, 297.6],
        [620.97, 321.96],
    ],
    "entry_zone_10": [
        [1171.5, 368.1],
        [1204.29, 393.82],
        [1196.8, 340.9],
        [1169.3, 323.1],
    ],
    "entry_zone_11": [
        [1137.0, 349.8],
        [1168.9, 369.0],
        [1166.4, 321.1],
        [1138.4, 303.7],
    ],
    "entry_zone_12": [
        [1103.2, 328.7],
        [1134.3, 348.4],
        [1135.8, 301.2],
        [1109.4, 283.7],
    ],
    "entry_zone_13": [
        [1069.4, 307.6],
        [1101.2, 327.0],
        [1107.2, 282.2],
        [1083.7, 267.9],
    ],
    "entry_zone_14": [[771.3, 371.5], [810.9, 341.4], [699.2, 273.1], [663.2, 294.6]],
}

zones = WEST_ZONES


# zones = NORTH_ZONES
# detection_zones = {key:zones[key] for key in zones if "entry_zone" in key}
# zone_polygons = {name: Polygon(coords) for name, coords in zones.items()}
# zone_centers = {
#     name: np.mean(coords, axis=0).astype(int) for name, coords in detection_zones.items()
# }


def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2.x) ** 2 + (point1[1] - point2.y) ** 2)


# # Initialize a dictionary to track detections per zone to ensure one blink per car per zone
# detected_cars_in_zone = {
#     zone: set() for zone in zone_centers
# }  # Each zone keeps track of detected cars

# NOTE: for online inferencing, implement clean up for car_zone_tracker, otherwise memory will run out
car_zone_tracker = {}  # Dictionary to keep track of the last blinked zone per car

# Method: Bounding Box Center Proximity Check with Zone Center (only one zone per car)
def check_zone_center_proximity(img, center_coords, zone_centers, car_id, proximity_threshold=20):
    # Check if car has already blinked a zone this frame
    #if car_id in car_zone_tracker:
    #    return None  # Skip if already blinked one zone

    for name, zone_center in zone_centers.items():
        # Calculate the distance from the car's center to the zone center
        distance = euclidean_distance(center_coords, zone_center)
        #print(f"[DEBUG] Car ID {car_id} - Distance to Zone '{name}' center: {distance:.2f} pixels")  # Debugging distance check

        # Check if the car is within the proximity threshold
        if distance <= proximity_threshold:
            if car_id not in detected_cars_in_zone[name]:  # Blink only if car hasn't triggered this zone
                #print(f"[DEBUG] Car ID {car_id} is within {distance:.2f} pixels of zone {name} center; blinking zone.")
                detected_cars_in_zone[name][car_id] = 1  # Track car as detected in this zone
                car_zone_tracker[car_id] = name  # Track the zone to limit blinking to one zone per car
                if len(car_zone_tracker) >= 100000:
                    car_zone_tracker.clear()
                return name  # Return the zone name to trigger blinking
            elif detected_cars_in_zone[name][car_id] >= 1 and detected_cars_in_zone[name][car_id] <= 5:
                print('entered')
                detected_cars_in_zone[name][car_id] += 1
                return name

    return None

def check_zone_proximity(
    img, bboxes, zones, car_id, zone_centers
):
    """
    Check if the bounding box center of a car is within any of the defined zones
    and if the zone's center point is within the bounding box.
    Ensures each car triggers only one zone blink and only once per zone.

    Parameters:
    - img: The image frame (not used here, included for compatibility).
    - bboxes: The bounding box coordinates of the car (x1, y1, x2, y2).
    - zones: Dictionary where keys are zone names and values are shapely.Polygon objects.
    - car_id: ID of the car.
    - zone_centers: Dictionary where keys are zone names and values are shapely.Point objects.

    Returns:
    - The zone name if both conditions are met; otherwise, None.
    """
    # Check if the car has already blinked a zone this frame
    #if car_id in car_zone_tracker:
    #    return None  # Skip if already blinked one zone

    # Calculate the center of the car's bounding box
    center_coords = (
        int(bboxes[0] + (bboxes[2] - bboxes[0]) / 2),
        int(bboxes[1] + (bboxes[3] - bboxes[1]) / 2)
    )
    car_center = Point(center_coords)  # Convert car center to a Shapely Point

    for name, zone_polygon in zones.items():
        # Get the zone center point
        zone_center = zone_centers[name]

        # Check if the car's center is within the zone polygon
        if car_center.within(zone_polygon):
            # Check if the zone's center point is within the car's bounding box
            if (
                bboxes[0] <= zone_center.x <= bboxes[2] and
                bboxes[1] <= zone_center.y <= bboxes[3]
            ):
                if car_id not in detected_cars_in_zone[name]:  # Blink only if not already triggered
                    #if opt.debug:
                        #print(f"[DEBUG] Car ID {car_id} is within zone '{name}' and zone center is within the car's bounding box; blinking zone.")
                    detected_cars_in_zone[name][car_id] = 1  # Track car as detected in this zone
                    car_zone_tracker[car_id] = name  # Track the zone to limit blinking
                    if len(car_zone_tracker) >= 100000:
                        car_zone_tracker.clear()
                    return name  # Return the zone name to trigger blinking
                elif detected_cars_in_zone[name][car_id] >= 1 and detected_cars_in_zone[name][car_id] <= 5:
                    print('entered')
                    detected_cars_in_zone[name][car_id] += 1
                    return name

    return None
# Zone highlighting helper function to display zones and blink if needed
def highlight_zone(img, zone_name, color=(255, 0, 0), blink=False):
    """
    Draws the zone outline on the image.
    - Default color is blue; changes to red when blinking.
    """
    color = (0, 0, 255) if blink else color  # Red if blinking, blue otherwise
    points = np.array(zones[zone_name], np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [points], isClosed=True, color=color, thickness=2)


def detect(save_img=False, line_thickness=1):
    show_vid, save_txt, imgsz, trace = (
        opt.show_vid,
        opt.save_txt,
        opt.img_size,
        opt.trace,
    )
    source = 'rtsp://wowza01.bellevuewa.gov:1935/live/CCTV047W.stream'
    weights = 'weights/yolov7-e6e.pt'
    save_img = not opt.nosave and not source.endswith(".txt")  # save inference images
    webcam = 'rtsp://wowza01.bellevuewa.gov:1935/live/CCTV047W.stream'
    save_crop = True
    project = ROOT / "runs/track"  # save results to project/name
    exp_name = "exp"  # save results to project/name
    strong_sort_weights = WEIGHTS / "osnet_x0_25_msmt17.pt"  # model.pt path,
    config_strongsort = ROOT / "strong_sort/configs/strong_sort.yaml"
    save_txt = opt.save_txt  # save results to *.txt
    save_conf = opt.save_conf  # save confidences in --save-txt labels
    hide_labels = opt.hide_labels  # hide labels
    hide_conf = opt.hide_conf  # hide confidences
    hide_class = opt.hide_class  # hide IDs
    count = opt.count
    save_vid = opt.save_vid
    save_img = opt.save_img
    line_thickness = opt.line_thickness
    draw = opt.draw

    # Directories
    save_dir = Path(
        increment_path(Path(opt.project) / opt.exp_name, exist_ok=opt.exist_ok)
    )  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(
        parents=True, exist_ok=True
    )  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != "cpu"  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    # imgsz = 320
    # print("imgsz: ",imgsz)

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name="resnet101", n=2)  # initialize
        modelc.load_state_dict(
            torch.load("weights/resnet101.pt", map_location=device)["model"]
        ).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        # view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        nr_sources = 1
    vid_path, vid_writer, txt_path = (
        [None] * nr_sources,
        [None] * nr_sources,
        [None] * nr_sources,
    )

    # initialize StrongSORT
    cfg = get_config()
    cfg.merge_from_file(opt.config_strongsort)

    # Create as many strong sort instances as there are video sources
    strongsort_list = []
    for i in range(nr_sources):
        strongsort_list.append(
            StrongSORT(
                strong_sort_weights,
                device,
                max_dist=cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT.MAX_AGE,
                n_init=cfg.STRONGSORT.N_INIT,
                nn_budget=cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT.EMA_ALPHA,
            )
        )
    outputs = [None] * nr_sources

    trajectory = {}
    # Get names and colors
    names = model.module.names if hasattr(model, "module") else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != "cpu":
        model(
            torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))
        )  # run once
    t0 = time.time()

    # Run tracking
    # model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources

    # for path, img, im0s, vid_cap in dataset:
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        print("frame_idx:", frame_idx)
        t1 = time_synchronized()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_synchronized()
        dt[0] += t2 - t1

        # Inference
        pred = model(img, augment=opt.augment)[0]
        t3 = time_synchronized()
        dt[1] += t3 - t2
        # Apply NMS
        pred = non_max_suppression(
            pred,
            opt.conf_thres,
            opt.iou_thres,
            classes=[0, 1, 2, 3, 5, 7],
            agnostic=True
        )
        dt[2] += time_synchronized() - t3

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], "%g: " % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, "", im0s, getattr(dataset, "frame", 0)
            
            # Draw zones in blue
            for zone_name in zones:
                highlight_zone(im0, zone_name, color=(255, 0, 0))
            
            curr_frames[i] = im0
            p = Path(p)  # to Path
            txt_file_name = p.name
            save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, .
            txt_path = str(save_dir / "labels" / p.stem)  # im.txt

            s += "%gx%g " % img.shape[2:]  # print string
            # imc = im0.copy() if save_crop else im0  # for save_crop

            if cfg.STRONGSORT.ECC:  # camera motion compensation
                strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # if len(det):
            if True:
                # Rescale boxes from img_size to im0 size
                # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape)

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to strongsort
                t4 = time_synchronized()
                outputs[i] = strongsort_list[i].update(
                    xywhs.cpu(), confs.cpu(), clss.cpu(), im0
                )
                t5 = time_synchronized()
                dt[3] += t5 - t4

                # draw boxes for visualization
                # if len(outputs[i]) > 0:
                # if len(outputs[i]) > -1:
                # Check if the car should trigger a zone blink
                # blink_zone_name = check_zone_center_proximity(
                #     im0, (x_center, y_center), zone_centers, car_id
                # )
                # if blink_zone_name:
                #     highlight_zone(im0, blink_zone_name, blink=True)
                #     break
                for j, (output, conf) in enumerate(zip(outputs[i], confs)):
                    # import pdb;pdb.set_trace()
                    bboxes = output[0:4]
                    # bboxes = [output[0],output[1],output[2]-output[0],output[3]-output[1]]
                    car_id = output[4] % 100000  # + 1317
                    cls = output[5]

                    x_center = int(bboxes[0] + (bboxes[2] - bboxes[0]) / 2)
                    y_center = int(bboxes[1] + (bboxes[3] - bboxes[1]) / 2)

                    # # Draw zones in blue
                    # for zone_name in zones:
                    #     highlight_zone(im0, zone_name, color=(255, 0, 0))
                    blink_zone_name = check_zone_center_proximity(
                         im0, (x_center, y_center), zone_centers, car_id
                    )
                    if not blink_zone_name:
                        blink_zone_name =  check_zone_proximity(
                            img, bboxes, detection_zone_polygons, car_id, zone_centers)
                    if blink_zone_name:
                        highlight_zone(im0, blink_zone_name, blink=True)

                    # Draw bounding box and center point
                    # cv2.rectangle(
                    #     im0,
                    #     (int(bboxes[0]), int(bboxes[1])),
                    #     (int(bboxes[2]), int(bboxes[3])),
                    #     (36, 255, 12),
                    #     2,
                    # )
                    # cv2.putText(
                    #     im0,
                    #     f"ID {int(car_id)}",
                    #     (int(bboxes[0]) - 10, int(bboxes[1]) - 15),
                    #     cv2.FONT_HERSHEY_SIMPLEX,
                    #     0.6,
                    #     (36, 255, 12),
                    #     2,
                    # )
                    cv2.circle(im0, (x_center, y_center), 5, (0, 0, 255), -1)

                    # # ROI - Counting Zone
                    # point = (int(bboxes[0])+int((output[2] - output[0])/2), int(bboxes[3]))
                    # bbox = ((0, 0), (277, 231))
                    # if not is_point_in_bbox(point, bbox):

                    # bbox_down = ((214, 212), (811, 465)) #Belleue1 down
                    # bbox_up = ((313,110), (962,335)) #Belleue1 up

                    # bbox_down = ((248, 211), (855, 430)) #Belleue2 down
                    # bbox_up = ((641,112), (1000,339)) #Belleue2 up

                    # bbox_down = ((187, 206), (675, 375)) #Belleue3 down
                    # bbox_up = ((602,153), (958,351)) #Belleue3 up

                    # bbox_down = ((293, 270), (800, 400)) #total Belleue down
                    # bbox_up = ((647, 20), (1000, 250)) #toal Belleue up

                    # speed####################################
                    # homography_matrix_obtained = [[-960.846498804670659, 1742.259116353276113, 258427.780339045682922],[-340.443283551905608, -392.117230790261431, -31685.871718625941867],[1.408962390454085, 0.557000179163394, 1.000000000000000]]
                    # mat=homography_matrix_obtained
                    # matinv=np.linalg.inv(mat)#.I

                    # bbox_left = output[0]
                    # bbox_top = output[1]
                    # bbox_w = output[2] - output[0]
                    # bbox_h = output[3] - output[1]
                    # c_x = bbox_left+bbox_w/2
                    # c_y = bbox_top+bbox_h/2
                    # fpoint = [c_x,c_y,1]
                    # hh=np.dot(matinv,fpoint)
                    # scalar=hh[2]
                    # c3d_x1 = hh[0]/scalar
                    # c3d_y1 = hh[1]/scalar
                    ##########################################

                    if draw:
                        # object trajectory
                        center = (
                            (int(bboxes[0]) + int(bboxes[2])) // 2,
                            (int(bboxes[1]) + int(bboxes[3])) // 2,
                        )
                        if car_id not in trajectory:
                            trajectory[car_id] = []
                        trajectory[car_id].append(center)
                        for i1 in range(1, len(trajectory[car_id])):
                            if (
                                trajectory[car_id][i1 - 1] is None
                                or trajectory[car_id][i1] is None
                            ):
                                continue
                            # thickness = int(np.sqrt(1000/float(i1+10))*0.3)
                            thickness = 2
                            try:
                                cv2.line(
                                    im0,
                                    trajectory[car_id][i1 - 1],
                                    trajectory[car_id][i1],
                                    (0, 0, 255),
                                    thickness,
                                )
                            except:
                                pass

                    if save_txt:
                        # if is_point_in_bbox(point, bbox_down) or is_point_in_bbox(point, bbox_up):
                        # to MOT format
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2] - output[0]
                        bbox_h = output[3] - output[1]
                        # Write MOT compliant results to file
                        with open(txt_path + ".txt", "a") as f:
                            f.write(
                                ("%g " * 11 + "\n")
                                % (
                                    frame_idx + 1,
                                    cls,
                                    car_id,
                                    bbox_left,  # MOT format
                                    bbox_top,
                                    bbox_w,
                                    bbox_h,
                                    -1,
                                    -1,
                                    -1,
                                    -1,
                                )
                            )

                        isExist = os.path.exists("tmg.csv")
                        if not isExist:
                            with open("tmg.csv", "w", newline="") as csvfile:
                                # fieldnames = ['first_name', 'last_name']
                                fieldnames = [
                                    "Direction of Travel",
                                    "Lane of Travel",
                                    "Year of Data",
                                    "Month of Year",
                                    "Day of Month",
                                    "Hour of Day",
                                    "Minute of Hour",
                                    "Second of Minute",
                                    "Sub Second of Second",
                                    "Vehicle of Speed",
                                    "Vehicle Classification",
                                    "Number of Axles",
                                ]
                                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                                writer.writeheader()

                        # Write TMG csv
                        with open("tmg.csv", "a", newline="") as csvfile:
                            # tmg_time = opt.stime

                            date_object = datetime.fromtimestamp(t0)
                            day = date_object.day
                            month = date_object.month
                            year = date_object.year
                            hour = date_object.hour
                            minute = date_object.minute
                            microsecond = date_object.microsecond
                            second = date_object.second
                            # import pdb;pdb.set_trace()

                            # fieldnames = ['first_name', 'last_name']
                            fieldnames = [
                                "Direction of Travel",
                                "Lane of Travel",
                                "Year of Data",
                                "Month of Year",
                                "Day of Month",
                                "Hour of Day",
                                "Minute of Hour",
                                "Second of Minute",
                                "Sub Second of Second",
                                "Vehicle of Speed",
                                "Vehicle Classification",
                                "Number of Axles",
                            ]
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                            # writer.writeheader()
                            # writer.writerow({'Direction of Travel': '1', 'Lane of Travel': '1','Year of Data':tmg_time[0],'Month of Year':tmg_time[1],'Day of Month':tmg_time[2],'Hour of Day':tmg_time[3],'Minute of Hour':tmg_time[4],'Second of Minute':'00','Sub Second of Second':'00','Vehicle of Speed':'xx','Vehicle Classification':'xx','Number of Axles':'2'})
                            writer.writerow(
                                {
                                    "Direction of Travel": "1",
                                    "Lane of Travel": "1",
                                    "Year of Data": year,
                                    "Month of Year": month,
                                    "Day of Month": day,
                                    "Hour of Day": hour,
                                    "Minute of Hour": minute,
                                    "Second of Minute": second,
                                    "Sub Second of Second": "00",
                                    "Vehicle of Speed": "xx",
                                    "Vehicle Classification": "xx",
                                    "Number of Axles": "2",
                                }
                            )

                            # writer.writerow({'first_name': 'Lovely', 'last_name': 'Spam'})
                            # writer.writerow({'first_name': 'Wonderful', 'last_name': 'Spam'})

                    if save_vid or save_crop or show_vid:  # Add bbox to image
                        c = int(cls)  # integer class
                        id = int(car_id)  # integer id
                        label = (
                            None
                            if hide_labels
                            else (
                                f"{id} {names[c]}"
                                if hide_conf
                                else (
                                    f"{id} {conf:.2f}"
                                    if hide_class
                                    else f"{id} {names[c]} {conf:.2f}"
                                )
                            )
                        )
                        plot_one_box(
                            bboxes,
                            im0,
                            label=label,
                            color=colors[int(cls)],
                            line_thickness=line_thickness,
                        )

                ### Print time (inference + NMS)
                print(f"{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)")

            else:
                strongsort_list[i].increment_ages()
                print("No detections")

            if count:
                itemDict = {}
                ## NOTE: this works only if save-txt is true
                try:
                    df = pd.read_csv(
                        txt_path + ".txt", header=None, delim_whitespace=True
                    )
                    df = df.iloc[:, 0:3]
                    df.columns = ["frameid", "class", "trackid"]
                    df = df[["class", "trackid"]]
                    df = (
                        df.groupby("trackid")["class"]
                        .apply(list)
                        .apply(lambda x: sorted(x))
                    ).reset_index()

                    df.colums = ["trackid", "class"]
                    df["class"] = df["class"].apply(
                        lambda x: Counter(x).most_common(1)[0][0]
                    )
                    vc = df["class"].value_counts()
                    vc = dict(vc)

                    vc2 = {}
                    for key, val in enumerate(names):
                        vc2[key] = val
                    itemDict = dict((vc2[key], value) for (key, value) in vc.items())
                    itemDict = dict(sorted(itemDict.items(), key=lambda item: item[0]))

                    # import pdb;pdb.set_trace()
                    del itemDict["tile"]
                    # print(itemDict)

                except:
                    pass

                if save_txt:
                    ## overlay
                    display = im0.copy()
                    h, w = im0.shape[0], im0.shape[1]
                    x1, y1, x2, y2 = 10, 10, 10, 70
                    txt_size = cv2.getTextSize(
                        str(itemDict), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                    )[0]
                    cv2.rectangle(
                        im0, (x1, y1 + 1), (txt_size[0] * 2, y2), (0, 0, 0), -1
                    )
                    # cv2.putText(im0, '{}'.format(itemDict), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX,0.7, (210, 210, 210), 2)
                    cv2.putText(
                        im0,
                        "{}".format(itemDict),
                        (x1 + 10, y1 + 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (210, 210, 210),
                        2,
                    )
                    cv2.addWeighted(im0, 0.7, display, 1 - 0.7, 0, im0)

            # current frame // tesing
            cv2.imwrite("testing.jpg", im0)

            # Stream results
            if show_vid:
                inf = f"{s}Done. ({t2 - t1:.3f}s)"
                # cv2.putText(im0, str(inf), (30,160), cv2.FONT_HERSHEY_SIMPLEX,0.7,(40,40,40),2)
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord("q"):  # q to quit
                    break

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        print("fps:", fps)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += ".mp4"
                    vid_writer = cv2.VideoWriter(
                        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                    )
                vid_writer.write(im0)
            try:
                if not frame_queue.full():
                    frame_queue.put(im0)
                    cv2.imwrite('test.jpg', im0)
            except:
                pass
            prev_frames[i] = curr_frames[i]

    if save_txt or save_vid or save_img:
        print(f"Results saved to ", save_dir)
    print(f"Done. ({time.time() - t0:.3f}s)")

def start_detection():
    """Start detection in a separate thread."""
    with torch.no_grad():
        detect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yolo-weights",
        nargs="+",
        type=str,
        default="weights/yolov7-tiny.pt",
        help="model.pt path(s)",
    )
    parser.add_argument(
        "--strong-sort-weights", type=str, default=WEIGHTS / "osnet_x0_25_msmt17.pt"
    )
    parser.add_argument(
        "--config-strongsort", type=str, default="strong_sort/configs/strong_sort.yaml"
    )
    parser.add_argument(
        "--source", type=str, default="inference/images", help="source"
    )  # file/folder, 0 for webcam
    parser.add_argument(
        "--img-size", type=int, default=640, help="inference size (pixels)"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.25, help="object confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.45, help="IOU threshold for NMS"
    )
    parser.add_argument(
        "--device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--show-vid", action="store_true", default=False, help="display results"
    )
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-img", action="store_true", help="save results to *.jpg")
    parser.add_argument(
        "--save-conf", action="store_true", help="save confidences in --save-txt labels"
    )
    parser.add_argument(
        "--nosave", action="store_true", default=True, help="do not save images/videos"
    )
    parser.add_argument(
        "--save-vid", action="store_true", help="save video tracking results"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="filter by class: --class 0, or --class 0 2 3",
    )
    parser.add_argument(
        "--agnostic-nms", action="store_true", help="class-agnostic NMS"
    )
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument(
        "--project", default="runs/track", help="save results to project/name"
    )
    parser.add_argument(
        "--exp-name", default="exp", help="save results to project/name"
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument("--trace", action="store_true", help="trace model")
    parser.add_argument(
        "--line-thickness", default=1, type=int, help="bounding box thickness (pixels)"
    )
    parser.add_argument(
        "--hide-labels", default=False, action="store_true", help="hide labels"
    )
    parser.add_argument(
        "--hide-conf", default=False, action="store_true", help="hide confidences"
    )
    parser.add_argument(
        "--hide-class", default=False, action="store_true", help="hide IDs"
    )
    parser.add_argument(
        "--count", action="store_true", help="display all MOT counts results on screen"
    )
    parser.add_argument(
        "--draw", action="store_true", help="display object trajectory lines"
    )
    parser.add_argument("--debug", action="store_true", help="if print out debug message")

    parser.add_argument(
        "--perspective",
        choices=["North", "West", "South"],
        help="perspective of camera"
    )

    # parser.add_argument('--stime', default='2024-1-1-11-00', action='store_true', help='setup the starting time of tracking')

    opt = parser.parse_args()
    detection_zones = {key:zones[key] for key in zones if "entry_zone" in key}
    detection_zone_polygons = {key: Polygon(zones[key]) for key in zones if "entry_zone" in key}
    zone_polygons = {name: Polygon(coords) for name, coords in zones.items()}
    zone_centers = {name: Point(np.mean(coords, axis=0).astype(int)) for name, coords in detection_zones.items()}
    #detected_cars_in_zone = {zone: set() for zone in zone_centers}
    detected_cars_in_zone = {zone: {} for zone in zone_centers}
    
    detection_thread = threading.Thread(target=start_detection)
    detection_thread.daemon = True
    detection_thread.start()

    # Run Flask application
    app.run(host='0.0.0.0', port=3004, threaded=True)