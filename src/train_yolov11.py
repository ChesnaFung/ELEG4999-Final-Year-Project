from ultralytics import YOLO

def main():
    # Load the YOLOv11s model
    model = YOLO("yolo11s.pt")

    # Start training
    model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        batch=128,
        device=[0, 1, 2, 3, 4, 5, 6, 7],  # uses 8 GPUs
        project="FYP_TurnSignal",
        name="v11_8GPU_Final",
        exist_ok=True
    )

if __name__ == '__main__':
    main()