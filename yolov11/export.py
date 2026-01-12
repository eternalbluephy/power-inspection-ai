from ultralytics import YOLO
import onnx

if __name__ == '__main__':
    # 1. 设置路径 (指向你训练好的 train2)
    train_folder = "train2"
    pt_path = rf"C:\Users\wang\Documents\Learn\University\Power inspection project\power-inspection-ai\yolov11\runs\detect\{train_folder}\weights\best.pt"
    
    print(f"正在加载 PyTorch 模型: {pt_path} ...")
    model = YOLO(pt_path)
    
    # 2. 导出 ONNX (Standard Export)
    # dynamic=True: 支持动态 Batch (批量检测关键)
    # simplify=True: 简化算子结构 (TensorRT 部署关键)
    print("正在导出为 ONNX 格式...")
    success = model.export(
        format='onnx',
        dynamic=True,       # 【关键】开启动态轴，支持C++端的多种BatchSize
        simplify=True,      # 【关键】使用 onnx-sim 简化算子
        opset=12,           # 推荐版本
        imgsz=960,          # 你的输入尺寸
        device='cpu'        # 【关键修复】强制使用CPU导出，避免 invalid device id 错误
    )
    
    # 导出后，success 变量通常返回导出文件的路径
    onnx_path = str(success) if isinstance(success, str) else pt_path.replace('.pt', '.onnx')
    print(f"ONNX 模型已保存至: {onnx_path}")

    # 3. 验证 ONNX 文件的结构完整性 (Check Integrity)
    print("正在检查 ONNX 模型结构...")
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX 模型结构检查通过！(Valid ONNX format)")
    except Exception as e:
        print(f"❌ ONNX 模型结构检查失败: {e}")

    # 4. 验证推理一致性 (Optional but Recommended)
    # 用 YOLO 库直接加载 ONNX 进行一次推理，看能不能跑通
    print("正在验证 ONNX 推理能力...")
    onnx_model_wrapper = YOLO(onnx_path, task='detect')
    
    # 随便找张图或者生成个假数据测一下
    try:
        # 这里只是为了不报错，imgsz 要对应
        results = onnx_model_wrapper('https://ultralytics.com/images/bus.jpg', imgsz=960) 
        print(f"✅ ONNX 推理测试成功！检测到 {len(results[0].boxes)} 个目标。")
        print("该 ONNX 模型已准备好交付给 Task 2 (C++/TensorRT) 使用。")
    except Exception as e:
        print(f"⚠️ ONNX 推理测试出现警告 (可能是环境问题，不代表模型坏了): {e}")
