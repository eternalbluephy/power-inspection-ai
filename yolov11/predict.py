import cv2
from ultralytics import YOLO
def hsv2bgr(h, s, v):
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if h_i == 0:
        r, g, b = v, t, p
    elif h_i == 1:
        r, g, b = q, v, p
    elif h_i == 2:
            r, g, b = p, v, t
    elif h_i == 3:
        r, g, b = p, q, v
    elif h_i == 4:
        r, g, b = t, p, v
    elif h_i == 5:
        r, g, b = v, p, q
    return int(b * 255), int(g * 255), int(r * 255)
def random_color(id):
    h_plane = (((id << 2) ^ 0x937151) % 100) / 100.0
    s_plane = (((id << 3) ^ 0x315793) % 100) / 100.0
    return hsv2bgr(h_plane, s_plane, 1)
if __name__ == "__main__":
    # --- 1. 参数设置 ---
    img_filename = "insulator_broken_001284.JPG"
    
    # 模型权重路径：训练完成后生成的最佳模型 best.pt
    # 请根据实际情况修改路径
    model_path = r"C:\Users\wang\Documents\Learn\University\Power inspection project\power-inspection-ai\yolov11\runs\detect\train\weights\best.pt"
    
    # 待检测图片路径
    img_path = fr"C:\Users\wang\Documents\Learn\University\Power inspection project\power-inspection-ai\yolov11\yolo_data\val\images\{img_filename}"

    # --- 2. 加载模型 ---
    print(f"正在加载模型: {model_path} ...")
    model = YOLO(model_path, task="detect")

    # --- 3. 读取图片 ---
    print(f"正在读取图片: {img_path} ...")
    img = cv2.imread(img_path)
    if img is None:
        print("错误: 无法读取图片，请检查路径。")
        exit()

    # --- 4. 执行推理 ---
    # conf=0.25 是默认置信度阈值，你可以修改它 (比如 conf=0.4) 来过滤低置信度目标
    results = model(img)[0]

    # --- 5. 解析结果 ---
    names = results.names  # 类别名称字典 {0: 'insulator', ...}
    boxes = results.boxes.data.tolist() # 获取边界框数据 [x1, y1, x2, y2, conf, cls]

    print(f"检测到 {len(boxes)} 个目标。")

    for obj in boxes:
        # 提取坐标 (左上角 x,y, 右下角 x,y)
        left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
        confidence = obj[4] # 置信度
        label = int(obj[5]) # 类别 ID
        
        # 获取随机颜色 (根据类别ID)
        color = random_color(label)
        
        # --- 6. 绘制结果 ---
        # 画矩形框
        cv2.rectangle(img, (left, top), (right, bottom), color=color, thickness=2, lineType=cv2.LINE_AA)
        
        # 准备标签文字 (类别名 + 置信度)
        caption = f"{names[label]} {confidence:.2f}"
        
        # 计算文字背景框大小
        w, h = cv2.getTextSize(caption, 0, 1, 2)[0]
        
        # 画文字背景 (实心矩形)
        cv2.rectangle(img, (left - 3, top - 33), (left + w + 10, top), color, -1)
        
        # 写文字 (白色)
        cv2.putText(img, caption, (left, top - 5), 0, 1, (0, 0, 0), 2, 16)

    # --- 7. 保存并显示 ---
    output_path = "predict.jpg"
    cv2.imwrite(output_path, img)
    print(f"检测完成，结果已保存至: {output_path}")