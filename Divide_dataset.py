import os
import random
import shutil
from tqdm import tqdm

# 1. 使用 r'' 防止路径转义问题，或者统一使用正斜杠 /
raw_image_dir = r'dataset\insulator\images'
raw_label_dir = r'dataset\insulator\labels'
output_dir = r'yolov11\yolo_data'
train_ratio = 0.8

# 2. 【关键】如果输出目录已存在，先删除，防止多次运行导致数据混杂/泄露
if os.path.exists(output_dir):
    print(f"正在清理旧目录: {output_dir}...")
    shutil.rmtree(output_dir)

# 创建目录结构
os.makedirs(os.path.join(output_dir, 'train', 'images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'train', 'labels'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'val', 'images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'val', 'labels'), exist_ok=True)

print("正在读取文件列表...")
# 支持更多格式，防止漏掉 png 等图片
valid_extensions = ('.jpg', '.JPG', '.png', '.PNG', '.jpeg', '.bmp')
image_filenames = [f for f in os.listdir(raw_image_dir) if f.endswith(valid_extensions)]

random.shuffle(image_filenames)

train_num = int(len(image_filenames) * train_ratio)
train_file = image_filenames[:train_num]
val_file = image_filenames[train_num:]

print(f"总计: {len(image_filenames)} 张 | 训练集: {len(train_file)} 张 | 验证集: {len(val_file)} 张")

# 定义一个辅助函数来处理复制，减少重复代码
def copy_files(file_list, target_subdir, desc_text):
    i=0
    for file in tqdm(file_list, desc=desc_text):
        # 复制图片
        shutil.copy(os.path.join(raw_image_dir, file), os.path.join(output_dir, target_subdir, 'images', file))
        
        # 处理标签
        label_file = os.path.splitext(file)[0] + '.txt'
        label_src = os.path.join(raw_label_dir, label_file)
        label_dst = os.path.join(output_dir, target_subdir, "labels", label_file)
        
        if os.path.exists(label_src):
            shutil.copy(label_src, label_dst)
        else:
            # 标签不存在 → 创建空文件 (作为负样本)
            i=i+1
            print(f"提示：图片 {file} 无对应标签，已生成空标签文件。")
            with open(label_dst, 'w', encoding='utf-8') as f:
                pass
            # 仅在调试时取消注释下面这行，避免刷屏
    print(f"{desc_text} 完成，缺失标签文件共计: {i} 个。")

# 执行复制
copy_files(train_file, 'train', '处理训练集')
copy_files(val_file, 'val',  '处理验证集')