from ultralytics import YOLO
import os

def main():
    # 1. 載入模型 (從yolov8s.pt 開始)
    model = YOLO('yolov8s.pt') 

    # 2. 開始訓練
    model.train(
        data='data.yaml', # 請替換成你實際的 yaml 路徑
        epochs=100,                         # 根據你的需求調整
        imgsz=640,
        batch=128,                           # 8張顯卡可以設大一點，例如 64 或 128
        device=[0, 1, 2, 3, 4, 5, 6, 7],    # 使用你上次提到的 8 顆 GPU
        project='runs/detect/FYP_TurnSignal',
        name='v8_run',           # 這是你的資料夾名稱
        exist_ok=True,                      # 如果資料夾已存在，直接覆寫而不報錯
        pretrained=True
    )

if __name__ == '__main__':
    main()