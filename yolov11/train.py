import os
from ultralytics import YOLO
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    # 加载官⽅预训练模型
    # 🔺 核心修改：从 n (Nano) 换成 s (Small) 模型，这是提升 mAP 的最强手段
    model = YOLO("yolo11s.pt", task="detect") 
    
    # 模型训练
    results = model.train(
        data="data.yaml", 
        # --- 1. 减少轮次 & 收紧早停 ---
        epochs=150,      # s模型收敛稍慢，给到 150 轮
        patience=30,     # 耐心稍加一点
        
        batch=16, 
        imgsz=640,       # 保持 640，防止过拟合
        
        # --- 🚀 速度优化 ---
        workers=4,       
        cache=False,     
        
        # --- 🎯 抗过拟合关键参数 ---
        dropout=0.0,     
        weight_decay=0.005, # 保持高权重衰减
        
        # --- 增强策略微调 ---
        augment=True,    
        degrees=10.0,      
        translate=0.1,     
        scale=0.5,         
        shear=0.0,         
        perspective=0.0005,
        flipud=0.0,        
        fliplr=0.5,        

        # --- 色彩调整 ---
        hsv_h=0.015,       
        hsv_s=0.7,         
        hsv_v=0.4,         

        # --- 高级增强 ---
        mosaic=1.0,        
        mixup=0.15,        # 🔺 微调：稍微加大 mixup，帮助 S 模型泛化
        copy_paste=0.3,    
        
        close_mosaic=20, 
        
        # --- 进阶策略 ---
        cos_lr=True,     
    )