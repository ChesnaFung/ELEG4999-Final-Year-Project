import os
import random
import shutil

# 基礎設定
# 因為你就在專案目錄執行，路徑設為當前目錄 '.'
images_src = 'images'
labels_src = 'labels'

# 定義分割比例 (80% Train, 10% Val, 10% Test)
train_ratio = 0.8
val_ratio = 0.1

# 建立子資料夾
for folder in ['train', 'val', 'test']:
    os.makedirs(os.path.join(images_src, folder), exist_ok=True)
    os.makedirs(os.path.join(labels_src, folder), exist_ok=True)

# 取得所有圖片（不含路徑和副檔名）
all_images = [f.rsplit('.', 1)[0] for f in os.listdir(images_src) if f.endswith(('.jpg', '.png', '.jpeg'))]
random.seed(42)
random.shuffle(all_images)

# 計算分割點
total = len(all_images)
num_train = int(total * train_ratio)
num_val = int(total * val_ratio)

splits = {
    'train': all_images[:num_train],
    'val': all_images[num_train:num_train+num_val],
    'test': all_images[num_train+num_val:]
}

# 開始搬移檔案
for split_name, file_list in splits.items():
    print(f"Processing {split_name} set ({len(file_list)} files)...")
    for name in file_list:
        # 搬移圖片 (支援多種副檔名)
        for ext in ['.jpg', '.png', '.jpeg']:
            img_file = name + ext
            src_img = os.path.join(images_src, img_file)
            if os.path.exists(src_img):
                shutil.move(src_img, os.path.join(images_src, split_name, img_file))
        
        # 搬移對應的標籤檔
        label_file = name + '.txt'
        src_label = os.path.join(labels_src, label_file)
        if os.path.exists(src_label):
            shutil.move(src_label, os.path.join(labels_src, split_name, label_file))

print("✅ Data splitting completed!")