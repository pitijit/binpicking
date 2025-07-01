import os
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# Load YOLOv8 segmentation model
model = YOLO("/weights/best.pt")

# Class names and known real-world widths (cm)
class_names = ['Right Clip', 'Left Clip', 'Front Clip', 'Back Clip']
REAL_WIDTHS = {
    'Right Clip': 2.5, 
    'Left Clip': 2.5,
    'Front Clip': 2.0,
    'Back Clip': 2.0,
}
known_dis_1 = 37  # for Left/Right
known_dis_2 = 37  # for Front/Back

roi_polygon = np.array([[55, 50], [55, 422], [560, 422], [566, 50]]) #selected ROI only use area
reference_dir = "/refimge"
test_dir = "/test"
output_dir = "/test_annotated"
depth_output_dir = "/refimge_depth"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(depth_output_dir, exist_ok=True)

MIN_MASK_AREA = 600
MIN_CONTOUR_AREA = 600 

widths_px_per_class = defaultdict(list)
focals_per_class = defaultdict(list)
depths_per_class = defaultdict(list)

print("Computing focal lengths from reference images...")

for ref_img in os.listdir(reference_dir):
    if not ref_img.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    image_path = os.path.join(reference_dir, ref_img)
    image = cv2.imread(image_path)

    roi_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(roi_mask, [roi_polygon], 255)

    result = model(image)[0]
    masks = result.masks.data if result.masks else None
    classes = result.boxes.cls if result.boxes else []

    if masks is None:
        continue

    for i, mask in enumerate(masks):
        class_id = int(classes[i])
        class_name = class_names[class_id]

        mask_area = np.count_nonzero(mask.cpu().numpy())
        if mask_area < MIN_MASK_AREA:
            continue

        mask_np = mask.cpu().numpy().astype("uint8") * 255
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        if roi_mask[cy, cx] == 0:
            continue

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect

        # Adjust for angle and size
        if w < h:
            w, h = h, w
            angle += 90
        width_px = w

        if width_px < 5:
            continue

        real_width = REAL_WIDTHS[class_name]
        known_distance = known_dis_1 if "Left" in class_name or "Right" in class_name else known_dis_2
        focal = (width_px * known_distance) / real_width

        widths_px_per_class[class_name].append(width_px)
        focals_per_class[class_name].append(focal)

        box = cv2.boxPoints(((cx, cy), (w, h), angle))
        box = np.intp(box)
        cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

        text = f"{class_name}: width_px={width_px:.2f}, focal={focal:.2f}"
        cv2.putText(image, text, (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.polylines(image, [roi_polygon], isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.imwrite(os.path.join(output_dir, ref_img), image)

# Compute average focal lengths
computed_focals = {}
print("\nSummary of computed focal lengths and widths:")
for class_name in class_names:
    widths = widths_px_per_class[class_name]
    focals = focals_per_class[class_name]
    if not focals:
        continue
    focals_np = np.array(focals)
    lower, upper = np.percentile(focals_np, [10, 90])
    filtered_focals = [f for f in focals if lower <= f <= upper]
    avg_f = np.mean(filtered_focals)
    computed_focals[class_name] = avg_f
    print(f"{class_name}:\n  Avg Focal: {avg_f:.2f} (Filtered {len(focals) - len(filtered_focals)} outliers)")

print("\nAnnotated reference images saved to:", output_dir)

def estimate_depth_for_image(image_path, save_path=None):
    image = cv2.imread(image_path)
    roi_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(roi_mask, [roi_polygon], 255)

    result = model(image)[0]
    masks = result.masks.data if result.masks else None
    classes = result.boxes.cls if result.boxes else []

    if masks is None:
        print("No masks detected in", image_path)
        return

    for i, mask in enumerate(masks):
        class_id = int(classes[i])
        class_name = class_names[class_id]

        mask_area = np.count_nonzero(mask.cpu().numpy())
        if mask_area < MIN_MASK_AREA:
            continue

        mask_np = mask.cpu().numpy().astype("uint8") * 255
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        if roi_mask[cy, cx] == 0:
            continue

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect
        if w < h:
            w, h = h, w
            angle += 90
        width_px = w

        real_width = REAL_WIDTHS[class_name]
        focal = computed_focals.get(class_name)
        if focal is None or width_px < 5:
            print(f"Skipping {class_name} due to missing focal or invalid width_px.")
            continue

        depth = ((real_width * focal) / width_px) * 10  # cm to mm
        depths_per_class[class_name].append(depth)

        box = cv2.boxPoints(((cx, cy), (w, h), angle))
        box = np.intp(box)
        cv2.drawContours(image, [box], 0, (255, 255, 0), 2)
        text = f"{class_name}:{depth:.2f} mm"
        cv2.putText(image, text, (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print(text)

    if save_path:
        cv2.imwrite(save_path, image)
        print("Saved:", save_path)

    return image

print("\nEstimating depth on new images...")
for img_file in os.listdir(test_dir):
    if not img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue
    img_path = os.path.join(test_dir, img_file)
    save_path = os.path.join(depth_output_dir, img_file)
    estimate_depth_for_image(img_path, save_path)

print("\nSummary: Average Estimated Depth per Class (in mm)")
for class_name in class_names:
    depths = depths_per_class[class_name]
    if depths:
        avg_depth = np.mean(depths)
        print(f"{class_name}: {avg_depth:.2f} mm  (from {len(depths)} detections)")
    else:
        print(f"{class_name}: No detections")

print("\nDepth estimation complete. Annotated images saved to:", depth_output_dir)
