import os
import xml.etree.ElementTree as ET
import shutil
import random
from tqdm import tqdm

# --- 配置 ---
DATASET_ROOT = r"dataset/insulator"  # 原始数据集根目录
OUTPUT_ROOT = r"yolov11/yolo_data_converted" # 输出目录 (避免覆盖现有 yolo_data)

# 类别映射：将 XML 中的类别名称映射到 YOLO ID
# classes.txt: 0: ring_shifted, 1: nest, 2: broken, 3: burn
CLASS_MAP = {
    'defect': 2,        # 假设 defect 文件夹下的 defect 标签对应 broken (请根据实际情况调整)
    'broken': 2,
    'burn': 3,
    'nest': 1,
    'ring_shifted': 0,
    # 'insulator': -1,  # 忽略 insulator 标签 (不在检测列表中)
}

TRAIN_RATIO = 0.8

def convert_box(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def process_dataset():
    images_dir = os.path.join(DATASET_ROOT, "images")
    labels_root = os.path.join(DATASET_ROOT, "labels")
    
    # 检查路径
    if not os.path.exists(images_dir):
        print(f"❌ 错误: 找不到图片目录 {images_dir}")
        return

    # 准备输出目录
    if os.path.exists(OUTPUT_ROOT):
        shutil.rmtree(OUTPUT_ROOT)
    
    os.makedirs(os.path.join(OUTPUT_ROOT, "train", "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, "train", "labels"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, "val", "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, "val", "labels"), exist_ok=True)

    # 获取所有图片
    all_images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(all_images)
    
    split_idx = int(len(all_images) * TRAIN_RATIO)
    train_imgs = all_images[:split_idx]
    val_imgs = all_images[split_idx:]
    
    print(f"找到 {len(all_images)} 张图片。训练集: {len(train_imgs)}, 验证集: {len(val_imgs)}")

    def process_batch(image_list, subset):
        for img_file in tqdm(image_list, desc=f"Processing {subset}"):
            file_id = os.path.splitext(img_file)[0]
            
            # --- 1. 复制图片 ---
            src_img_path = os.path.join(images_dir, img_file)
            dst_img_path = os.path.join(OUTPUT_ROOT, subset, "images", img_file)
            shutil.copy(src_img_path, dst_img_path)
            
            # --- 2. 查找并合并 XML 标签 ---
            # 标签可能分布在 labels/defect/, labels/insulator/ 等子文件夹中
            # 或者直接在 labels/ 中
            
            found_objects = []
            
            # 搜索所有 potential XMLs
            # 策略：扫描 labels 下的所有子目录，寻找 {file_id}.xml
            
            xml_files = []
            if os.path.exists(labels_root):
                # 递归查找 file_id.xml
                for root, dirs, files in os.walk(labels_root):
                    if f"{file_id}.xml" in files:
                        xml_files.append(os.path.join(root, f"{file_id}.xml"))
            
            # 如果没找到，尝试找 .txt (LabelImg YOLO 格式)
            # 这里先只处理 XML
            
            img_w, img_h = 0, 0
            
            for xml_file in xml_files:
                try:
                    tree = ET.parse(xml_file)
                    root = tree.getroot()
                    
                    # 获取图片尺寸 (只取第一次)
                    if img_w == 0:
                        size = root.find('size')
                        if size is not None:
                            img_w = int(size.find('width').text)
                            img_h = int(size.find('height').text)
                    
                    for obj in root.iter('object'):
                        cls_name = obj.find('name').text
                        if cls_name not in CLASS_MAP or CLASS_MAP[cls_name] == -1:
                            continue
                        
                        cls_id = CLASS_MAP[cls_name]
                        xmlbox = obj.find('bndbox')
                        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                        
                        # Normalize
                        if img_w > 0 and img_h > 0:
                            bb = convert_box((img_w, img_h), b)
                            found_objects.append(f"{cls_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}")
                            
                except Exception as e:
                    print(f"Error parsing {xml_file}: {e}")

            # --- 3. 写入标签文件 ---
            # 即使没有目标，也建议生成空文件（负样本）
            dst_label_path = os.path.join(OUTPUT_ROOT, subset, "labels", f"{file_id}.txt")
            with open(dst_label_path, "w") as f:
                f.write("\n".join(found_objects))

    process_batch(train_imgs, "train")
    process_batch(val_imgs, "val")

    print("\n✅ 数据转换完成！此脚本未覆盖原 train/val，而是生成了 'yolov11/yolo_data_converted'。")
    print("请检查 labels 是否正确，然后修改 data.yaml 的 path 指向新目录，或者手动合并数据。")

if __name__ == "__main__":
    process_dataset()
