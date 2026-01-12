from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket
from fastapi.responses import JSONResponse, FileResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uvicorn
import cv2
import numpy as np
import io
import sys
import os
import glob
import asyncio
from typing import Optional

# --- 路径乱炖修复 ---
# 确保能找到 service 目录下的模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from service.inspection_service import InspectionService

app = FastAPI(title="Power Inspection AI Backend", version="1.0")

# Initialize Service
inspection_service = InspectionService()
# Set MODEL_PATH for report endpoint usage
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'yolov11', 'yolo11n.pt'))

# --- CORS 跨域设置 ---
# 允许前端其实任何地址访问 (开发环境可以设为 "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"status": "ok", "message": "Power Inspection API is running"}

# --- 0. 获取可用模型列表 ---
@app.get("/api/models")
def get_models():
    """
    返回所有可用的模型列表
    """
    models = inspection_service.get_available_models()
    # 提取简单的名字列表供前端展示，或者返回完整信息
    return {"status": "success", "models": [m["name"] for m in models], "details": models}

# --- 1. 单图检测接口 ---
@app.post("/api/predict")
async def predict(
    file: Optional[UploadFile] = File(None), 
    image_path: Optional[str] = Form(None),
    conf: float = 0.4, 
    model_name: Optional[str] = Form(None),
    return_image: bool = Form(True)
):
    """
    单图上传检测 (支持并发处理)
    支持文件上传 (file) 或 本地路径 (image_path)
    """
    try:
        # 1. 获取图片数据
        contents = None
        if image_path:
            # 本地路径模式 (解决大文件传输慢的问题)
            if os.path.exists(image_path):
                # 使用 numpy 读取以支持中文路径
                # 将文件读取为 bytes，保持与 UploadFile 接口一致
                contents = np.fromfile(image_path, dtype=np.uint8).tobytes()
            else:
                 return JSONResponse(status_code=400, content={"status": "error", "message": f"File not found: {image_path}"})
        elif file:
            # 传统上传模式
            contents = await file.read()
        else:
            return JSONResponse(status_code=400, content={"status": "error", "message": "No file or image_path provided"})
        
        # 2. 核心处理逻辑放入线程池 (CPU bound)
        # 避免阻塞主事件循环，实现高并发请求处理
        output = await run_in_threadpool(
            inspection_service.process_request, 
            contents, 
            conf=conf, 
            model_name=model_name,
            return_image=return_image
        )
        
        results = output["results"]
        processed_base64 = output.get("processed_base64", "")
        
        # Update global MODEL_PATH (best effort)
        global MODEL_PATH
        MODEL_PATH = str(output.get("model_path", MODEL_PATH))

        return JSONResponse(content={
            "status": "success",
            "count": len(results),
            "results": results, 
            "image_base64": processed_base64,
            "model_used": os.path.basename(MODEL_PATH)
        })

    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

# --- 2. 批量检测接口 ---
@app.post("/api/predict_batch")
async def predict_batch(
    files: List[UploadFile] = File(None),
    image_paths: List[str] = Form(None),
    conf: float = 0.4, 
    model_name: Optional[str] = Form(None),
    return_image: bool = Form(True)
):
    """
    多图批量检测 (支持路径传输优化)
    """
    global MODEL_PATH
    results_list = []
    
    # CASE A: 通过本地路径 (优先)
    if image_paths:
        for fpath in image_paths:
             try:
                if os.path.exists(fpath):
                    contents = np.fromfile(fpath, dtype=np.uint8).tobytes()
                    output = await run_in_threadpool(
                        inspection_service.process_request,
                        contents,
                        conf=conf,
                        model_name=model_name,
                        return_image=return_image
                    )
                    results_list.append({
                        "filename": os.path.basename(fpath),
                        "count": len(output["results"]),
                        "results": output["results"],
                        "image_base64": output.get("processed_base64", "")
                    })
                    
                    # Update global model path
                    MODEL_PATH = str(output.get("model_path", MODEL_PATH))
                else:
                     results_list.append({"filename": fpath, "error": "File not found"})
             except Exception as e:
                results_list.append({"filename": fpath, "error": str(e)})

    # CASE B: 通过文件上传
    elif files:
        for file in files:
            try:
                contents = await file.read()
                # 复用 inspection_service 逻辑
                output = await run_in_threadpool(
                    inspection_service.process_request,
                    contents,
                    conf=conf,
                    model_name=model_name,
                    return_image=return_image
                )
                
                results_list.append({
                    "filename": file.filename,
                    "count": len(output["results"]),
                    "results": output["results"],
                    "image_base64": output.get("processed_base64", "")
                })
                
                # Update global model path from first successful result
                MODEL_PATH = str(output.get("model_path", MODEL_PATH))
                
            except Exception as e:
                results_list.append({
                    "filename": file.filename,
                    "error": str(e)
                })

    return JSONResponse(content={
        "status": "success", 
        "total_files": len(results_list),
        "batch_results": results_list
    })

# --- 3. 实时视频流检测 (WebSocket) ---
@app.websocket("/ws/detect_stream")
async def websocket_endpoint(websocket: WebSocket, conf: float = 0.4):
    await websocket.accept()
    try:
        while True:
            # 接收前端发送的帧数据 (Expect bytes or base64)
            data = await websocket.receive_bytes()
            
            # 使用现有服务处理每一帧
            # 注意: 视频流通常不需要每次都画图返回，但这取决于前端需求
            # 这里我们假设前端发送 jpg bytes，后端返回检测结果 json + 画好框的图(可选)
            
            output = await run_in_threadpool(
                 inspection_service.process_request,
                 data,
                 conf=conf
                 # 视频流为了速度一般不切换模型
            )
            
            # 回传结果
            await websocket.send_json({
                "results": output["results"],
                # 如果带宽允许，也可以传 base64 图片回去实时显示
                "image_base64": output.get("processed_base64", "")
            })
            
    except Exception as e:
        print(f"WebSocket Error: {e}")
    finally:
         await websocket.close()

# --- 4. 获取训练报告图表 ---
@app.get("/api/report/{report_type}")
async def get_report(report_type: str, model_name: Optional[str] = None):
    """
    获取训练过程中的图表
    report_type: 'loss', 'confusion_matrix', 'pr_curve'
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yolo_root = os.path.join(project_root, 'yolov11')
    
    candidate_dirs = []

    # 策略 0: 如果指定了模型名，尝试解析其路径
    current_path = MODEL_PATH
    if model_name:
        # 使用 service 的逻辑查找 path
        available = inspection_service.get_available_models()
        for m in available:
            if m["name"] == model_name:
                current_path = m["path"]
                break
    
    # 策略 1: 尝试从当前 (或指定) MODEL_PATH 推断
    # 典型路径: .../runs/detect/train2/weights/best.pt -> .../runs/detect/train2/
    if "runs" in current_path and "weights" in current_path:
        # 回退两级: weights/ -> trainX/
        run_dir = os.path.dirname(os.path.dirname(current_path))
        candidate_dirs.append(run_dir)
        
    # 策略 2: 自动寻找最新的 runs/detect/train* 目录
        
    # 策略 2: 自动寻找最新的 runs/detect/train* 目录
    detect_runs = os.path.join(yolo_root, "runs", "detect")
    if os.path.exists(detect_runs):
        # 获取所有 train 目录，按修改时间排序
        subdirs = [os.path.join(detect_runs, d) for d in os.listdir(detect_runs) 
                   if os.path.isdir(os.path.join(detect_runs, d)) and d.startswith("train")]
        if subdirs:
            # 按时间倒序，最新的在前面
            subdirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            candidate_dirs.extend(subdirs)
            
    # 策略 3: 作为托底，允许在 yolo_root 直接查找 (不太可能，但也加上)
    candidate_dirs.append(yolo_root)

    # 目标文件名映射
    target_filenames = []
    if report_type == "loss":
        target_filenames = ["results.png"]
    elif report_type == "confusion_matrix":
        target_filenames = ["confusion_matrix_normalized.png", "confusion_matrix.png"]
    elif report_type == "pr_curve":
        target_filenames = ["BoxPR_curve.png", "P_curve.png", "R_curve.png"]
    else:
         raise HTTPException(status_code=400, detail="Unknown report type")

    # 遍历所有候选目录，寻找目标文件
    for base_dir in candidate_dirs:
        for fname in target_filenames:
            file_path = os.path.join(base_dir, fname)
            if os.path.exists(file_path):
                return FileResponse(file_path)
        
        # 如果精确匹配失败，尝试模糊匹配 (glob)
        # 例如 *confusion_matrix*.png
        pattern = os.path.join(base_dir, f"*{report_type}*.png")
        found = glob.glob(pattern)
        if found:
            # 优先返回看起来像图表的
            return FileResponse(found[0])

    raise HTTPException(status_code=404, detail=f"Report chart for '{report_type}' not found in specific runs or root.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)