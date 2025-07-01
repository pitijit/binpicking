# 2D vision-based bin-picking with monocular depth estimation
Solving bin picking problem using 2D vision, yolov8 instance segmentation and MiDaS depth estimation
![Screenshot (1458)](https://github.com/user-attachments/assets/0e60328b-86e3-4a85-ae34-256fb0e3f246)

## Monocular depth estimation
Monocular depth estimation is a computer vision task where an AI model tries to predict the depth information of a scene from a single image.In this project we use MiDaS model developed by Intel Labs. It estimates relative depth from a single RGB image, meaning it does not require stereo input or depth sensors.


### Result visualization
![Screenshot (1459)](https://github.com/user-attachments/assets/41afcbfd-69c5-4ea9-a62b-2edda80c0ea1)


## YOLOv8 instance segmentation
YOLOv8 Instance Segmentation is the version of YOLOv8 that detects and segments objects at the instance level—each object has its own mask.

### Dataset Structure
``` 
datasets/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```
Convert MS COCO to YOLO 
```
git clone https://github.com/Koldim2001/COCO_to_YOLOv8.git
cd COCO_to_YOLOv8
pip install -r requirements.txt
```

### Result visualization
![Screenshot (1462)](https://github.com/user-attachments/assets/f8e22d51-8d06-475f-88df-f3f83c17d916)


## Integration
### Result visualization
![Screenshot (1460)](https://github.com/user-attachments/assets/12208a40-82a9-45f5-84a2-39dc69e33c6b)
