import cv2
import torch
import numpy as np
import os
import time
import pyrealsense2 as rs
from ultralytics import YOLO
import socket
import math
import pickle

# Robot settings
ROBOT_IP = "xxx.xxx.xxx.x"   #ip address of robot controller
PORT = 5000

CALIBRATION_FILE = '/calibration_data.pkl'  #camera calibration file

# Constants
CONFIDENCE_THRESHOLD = 0.7
MIN_MASK_AREA = 600  #Selected only fully object
MIN_CONTOUR_AREA = 600
scale_xy = 0.595  # mm/px  scale for x and y in realworld to image

device = "cuda" if torch.cuda.is_available() else "cpu" # I run by using cpu, can change to gpu

# Load MiDaS
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
midas.to(device).eval()
transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

# Load YOLOv8
model = YOLO("/weights/best.pt") # trained yolo model

# Save folder
save_folder = "images"
os.makedirs(save_folder, exist_ok=True)

# Load calibration data
with open(CALIBRATION_FILE, 'rb') as f:
    calibration_data = pickle.load(f)

mtx = calibration_data['camera_matrix']
dist = calibration_data['distortion_coefficients']

# RealSense config
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)
align = rs.align(rs.stream.color)

width, height = 640, 480 #resolution

# Undistortion maps
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (width, height), cv2.CV_32FC1)

# Set manual exposure
profile = pipeline.get_active_profile()
sensor = profile.get_device().query_sensors()[1]
sensor.set_option(rs.option.enable_auto_exposure, False)
sensor.set_option(rs.option.exposure, 300)

# Connect socket to robot
def connect_robot(ip, port):
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((ip, port))
        print("Connected to ABB GoFa.")
        return client
    except socket.error as e:
        print(f"Socket connection error: {e}")
        return None
    
client = connect_robot(ROBOT_IP, PORT)

print("Press 'C' to capture an image. Press 'Q' to quit.")
correct_distortion = True

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        if correct_distortion:
            undistorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = roi
            undistorted = undistorted[y:y+h, x:x+w]
            undistorted = cv2.resize(undistorted, (width, height))
            cv2.imshow('', undistorted)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        if key == ord('c'):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(save_folder, f"image_{timestamp}.jpg")
            cv2.imwrite(filename, undistorted)
            print(f"Image saved: {filename}")

            img_rgb = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
            roi_polygon = np.array([[55, 50], [55, 422], [560, 422], [566, 50]])

            # Compute bounding rectangle
            x, y, w, h = cv2.boundingRect(roi_polygon)

            # Optional: Draw the ROI on the image for visualization
            cv2.polylines(undistorted, [roi_polygon], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.imshow("Auto-selected ROI", undistorted)
            cv2.waitKey(500)  # Display briefly
            cv2.destroyAllWindows()

            roi_rgb = img_rgb[y:y+h, x:x+w]
            input_batch = transform(roi_rgb).to(device)

            with torch.no_grad():
                prediction = midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1), size=roi_rgb.shape[:2],
                    mode="bicubic", align_corners=False
                ).squeeze()

            depth_map = prediction.cpu().numpy()
            depth_map_norm = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
            depth_map_display = (depth_map_norm * 255).astype(np.uint8)
            depth_map_display = cv2.applyColorMap(depth_map_display, cv2.COLORMAP_MAGMA)
            _, _, _, max_loc = cv2.minMaxLoc(depth_map_norm)
            closest_point = (max_loc[0] + x, max_loc[1] + y)

            results = model(undistorted)
            
            min_distance = float('inf')
            closest_object = None

            for r in results:
                if r.masks is None:
                    print("No masks found.")
                    continue

                boxes = r.boxes.xyxy.cpu().numpy()
                masks = r.masks.data.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()

                for j, mask in enumerate(masks):
                    if confs[j] < CONFIDENCE_THRESHOLD or np.count_nonzero(mask) < MIN_MASK_AREA:
                        continue

                    binary_mask = (mask * 255).astype('uint8')
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for cnt in contours:
                        if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
                            continue

                        rect = cv2.minAreaRect(cnt)
                        box = cv2.boxPoints(rect).astype(np.int32)
                        center, (w, h), angle = rect
                        center_point = (int(center[0]), int(center[1]))
                        distance = np.linalg.norm(np.array(center_point) - np.array(closest_point))

                        if distance < min_distance:
                            min_distance = distance
                            closest_object = (box, center, w, h, angle, confs[j], classes[j])

            if closest_object:
                box, center, w, h, angle, confidence, class_id = closest_object
                image_with_boxes = undistorted.copy()
                cv2.drawContours(image_with_boxes, [box], -1, (0, 255, 255), 2)

                if w > h:
                    x_axis_length = int(w // 2)
                    y_axis_length = int(h // 2)
                    angle_x = angle
                else:
                    x_axis_length = int(h // 2)
                    y_axis_length = int(w // 2)
                    angle_x = angle + 90

                angle_x = (angle_x + 180) % 360 - 180
                angle_y = angle_x + 90

                x_axis_start = (
                    int(center[0] - x_axis_length * np.cos(np.deg2rad(angle_x))),
                    int(center[1] - x_axis_length * np.sin(np.deg2rad(angle_x))),
                )
                x_axis_end = (
                    int(center[0] + x_axis_length * np.cos(np.deg2rad(angle_x))),
                    int(center[1] + x_axis_length * np.sin(np.deg2rad(angle_x))),
                )
                y_axis_start = (
                    int(center[0] - y_axis_length * np.cos(np.deg2rad(angle_y))),
                    int(center[1] - y_axis_length * np.sin(np.deg2rad(angle_y))),
                )
                y_axis_end = (
                    int(center[0] + y_axis_length * np.cos(np.deg2rad(angle_y))),
                    int(center[1] + y_axis_length * np.sin(np.deg2rad(angle_y))),
                )

                cv2.line(image_with_boxes, x_axis_start, x_axis_end, (255, 0, 0), 2)
                cv2.line(image_with_boxes, y_axis_start, y_axis_end, (0, 255, 0), 2)
                midpoint_x_mm = center[0] 
                midpoint_y_mm = center[1] 

                distance = depth_frame.get_distance(int(midpoint_x_mm), int(midpoint_y_mm))
                distance_mm = distance*1000  #convert to mm
                    
                print(f"Midpoint in mm: X = {midpoint_x_mm:.2f}, Y = {midpoint_y_mm:.2f}")
                print(f"Orientation angle: {angle_x:.2f} deg")
                print(f"Distance: {distance_mm :.2f} mm")

                label = f"Class {int(class_id)} ({confidence:.2f})" #show label on image
                cv2.putText(image_with_boxes, label, (box[0][0], box[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(image_with_boxes, f"Angle: {angle_x:.2f}", (int(center[0]), int(center[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                rot_rad = math.radians(-180 + angle_y)
                rotq2 = round(math.sin(rot_rad / 2), 4)
                rotq3 = round(math.cos(rot_rad / 2), 4)
                print(f"rotation q2 = {rotq2:.4f}, q3 = {rotq3:.4f}")

                target_x = round(549.40 + (379 - midpoint_y_mm) * scale_xy, 2)
                target_y = round(-89.15 + (527 - midpoint_x_mm) * scale_xy, 2)
                print(f"target in mm: X = {target_x:.2f}, Y = {target_y:.2f}")

                robtarget_data = {
                    "label": [int(class_id)],
                    "trans": [target_x, target_y, distance_mm],
                    "rot": [0, rotq2, rotq3, 0],
                    "conf": [0, 0, 0, 0],
                    "ext": [0, 0, 0, 0, 0, 0]
                }
                robtarget_str = ",".join(map(str, robtarget_data["label"] + robtarget_data["trans"] + robtarget_data["rot"] +
                                             robtarget_data["conf"] + robtarget_data["ext"]))

                client.sendall(robtarget_str.encode()) #send data to robot via socket
                start_time = time.time()
                print(f"Sent: {robtarget_str}")
                response = client.recv(1024).decode()
                end_time = time.time()
                print(f"Robot Response: {response}")
                latency_ms = (end_time - start_time)
                print(f"Latency: {latency_ms:.2f} ms")
                cv2.circle(image_with_boxes, closest_point, 5, (0, 0, 255), -1)
                cv2.imshow("Result", image_with_boxes)
                cv2.imshow("Depth Map", depth_map_display)

except Exception as e:
    print(f"Exception: {e}")
finally:
    pipeline.stop()
    if client:
        client.close()
    cv2.destroyAllWindows()
