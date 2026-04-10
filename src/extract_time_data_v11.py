from ultralytics import YOLO
import time
import numpy as np

# 1. 加載模型
model = YOLO('runs/detect/FYP_TurnSignal/v11_run/weights/best.pt')

# 2. 推薦設定（提升可重現性與速度）
source = 'images/test'          # 或指定一個包含多張測試圖片的資料夾
imgsz = 640
device = 0                      # GPU 0

# 3. 先做 Warmup（很重要！避免第一次 inference 特別慢）
print("Performing warmup...")
_ = model.predict(source='images/test', imgsz=imgsz, device=device, 
                  verbose=False, half=True, max_det=100)  # half=True 可加速（如果模型支援）

# 4. 正式測量（建議測 200~500 張圖，取平均）
num_runs = 5                    # 重複跑幾次取平均，更穩定
inference_times = []
postprocess_times = []
total_times = []

print(f"Starting inference measurement on {source} ...")

for run in range(num_runs):
    start_time = time.time()
    
    results = model.predict(
        source=source,
        imgsz=imgsz,
        device=device,
        verbose=False,      # 關閉輸出文字
        half=True,          # FP16 加速（推薦）
        max_det=100,        # turn signal 通常不會太多
        stream=False        # False 時一次載入所有結果
    )
    
    # 收集所有圖片的 speed（避免只看第一張）
    for r in results:
        speed = r.speed
        inference_times.append(speed['inference'])
        postprocess_times.append(speed['postprocess'])
        total_times.append(speed['inference'] + speed['postprocess'])
    
    print(f"Run {run+1}/{num_runs} completed.")

# 5. 計算統計值
avg_inference = np.mean(inference_times)
avg_postprocess = np.mean(postprocess_times)
avg_total = np.mean(total_times)

fps_inference = 1000 / avg_inference
fps_total = 1000 / avg_total

print("\n=== Final Speed Results (averaged over all images and runs) ===")
print(f"Average Preprocess : {np.mean([r.speed['preprocess'] for r in results]):.2f} ms")
print(f"Average Inference  : {avg_inference:.2f} ms")
print(f"Average Postprocess: {avg_postprocess:.2f} ms")
print(f"Average Total Time : {avg_total:.2f} ms")
print(f"FPS (Inference only): {fps_inference:.1f}")
print(f"FPS (Inference + Postprocess): {fps_total:.1f}   ← 這是最常用在論文的數字")

# 如果你想跟之前寫的 9.1 ms / 110 FPS 對齊，通常用 Inference + Postprocess 作為 latency