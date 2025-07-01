from ultralytics import YOLO

if __name__ == "__main__":
    # Load the YOLOv8 segmentation model
    model = YOLO("yolov8s-seg.pt")  

    # Train the model on CPU
    model.train(
        data="/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0001,
        dropout=0.2,
        mosaic=1.0,
        mixup=0.5,
        degrees=10,
        translate=0.1,
        scale=0.5,
        flipud=0.5,
        fliplr=0.5,
        patience=20,
        device="cpu"  # <- explicitly specify CPU
    )