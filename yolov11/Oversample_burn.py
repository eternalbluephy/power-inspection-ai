import os
import shutil
from tqdm import tqdm

# 配置路径
data_root = r"yolov11/yolo_data/train"
img_dir = os.path.join(data_root, "images")
lbl_dir = os.path.join(data_root, "labels")

# 你的 burn 类别 ID (请检查 labels/classes.txt)
TARGET_CLASS_ID = 3  # 假设 burn 是 2，请修改！
COPY_TIMES = 5       # 复制几份？

print("正在扫描含 Burn 的样本...")
aug_count = 0

for label_file in tqdm(os.listdir(lbl_dir)):
    if not label_file.endswith(".txt"): continue
    
    src_lbl_path = os.path.join(lbl_dir, label_file)
    
    # 检查内容是否包含 burn
    has_target = False
    with open(src_lbl_path, 'r') as f:
        for line in f:
            if line.strip().startswith(f"{TARGET_CLASS_ID} "):
                has_target = True
                break
    
    if has_target:
        # 找到对应的图片
        basename = os.path.splitext(label_file)[0]
        # 尝试找 jpg/png
        found_img = None
        for ext in ['.jpg', '.JPG', '.png', '.PNG']:
            src_img_path = os.path.join(img_dir, basename + ext)
            if os.path.exists(src_img_path):
                found_img = src_img_path
                img_ext = ext
                break
        
        if found_img:
            # 开始复制
            for i in range(COPY_TIMES):
                new_basename = f"{basename}_aug{i}"
                
                # 复制图片
                dst_img = os.path.join(img_dir, new_basename + img_ext)
                shutil.copy(found_img, dst_img)
                
                # 复制标签
                dst_lbl = os.path.join(lbl_dir, new_basename + ".txt")
                shutil.copy(src_lbl_path, dst_lbl)
                
            aug_count += 1

print(f"完成！共对 {aug_count} 张含 Burn 的图片进行了 {COPY_TIMES} 倍增殖。")
print("请重新运行 train.py，模型会更加重视这些样本。")