import os
# --- 修复 OMP: Error #15 (防止重复加载 OpenMP 报错) ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 获取用户输入的训练目录
    print("-" * 30)
    default_train = "train"
    train_name = input(f"请输入训练文件夹名称 (例如 train, train2... 回车默认 '{default_train}'): ").strip()
    if not train_name:
        train_name = default_train

    # 自动构建路径：当前脚本所在目录/runs/detect/{train_name}/weights/best.pt
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "runs", "detect", train_name, "weights", "best.pt")
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"\n❌ 错误: 找不到模型文件！")
        print(f"路径: {model_path}")
        print("请检查文件夹名称输入是否正确 (不用输入路径，只需输入 'train2' 这种文件夹名)")
        exit(1)
        
    print(f"\n✅ 正在加载模型: {model_path}...")
    model = YOLO(model_path)

    # 2. 在验证集上评估模型
    # data="data.yaml" 会告诉它验证集图片和标签在哪
    # split='val' 表示使用验证集
    metrics = model.val(data="data.yaml", split='val')

    # 3. 打印核心指标
    print(f"mAP50 (平均精度): {metrics.box.map50}")
    print(f"mAP50-95 (高标准精度): {metrics.box.map}")
    print(f"Precision (查准率): {metrics.box.mp}")
    print(f"Recall (查全率): {metrics.box.mr}")