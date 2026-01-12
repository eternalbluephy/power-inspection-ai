import gradio as gr
import requests
import cv2
import numpy as np
import base64
import json
import time
from PIL import ImageGrab
import tkinter as tk
import os
import uuid

# 后端服务地址
API_BASE = "http://localhost:8000"
API_URL = f"{API_BASE}/api/predict"
BATCH_URL = f"{API_BASE}/api/predict_batch"
MODELS_URL = f"{API_BASE}/api/models"
REPORT_URL = f"{API_BASE}/api/report"

# 临时图片目录
TEMP_DIR = os.path.join(os.getcwd(), "temp_upload")
os.makedirs(TEMP_DIR, exist_ok=True)

# 全局 Session 复用连接
global_session = requests.Session()

def get_available_models():
    """从后端获取可用模型列表"""
    try:
        resp = requests.get(MODELS_URL, timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("models", [])
    except:
        return ["yolo11n.pt"] 
    return ["yolo11n.pt"]

# --- 屏幕区域选择器 (Tkinter Overlay) ---
class ScreenSelector:
    def __init__(self):
        self.root = None
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.selection = None # (x1, y1, x2, y2)

    def select_area(self):
        self.selection = None
        self.root = tk.Tk()
        # 全屏透明遮罩
        self.root.attributes('-alpha', 0.3)
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-topmost', True)
        self.root.config(bg='black')
        self.root.config(cursor="cross")
        
        # Canvas
        self.canvas = tk.Canvas(self.root, cursor="cross", bg="black")
        self.canvas.pack(fill="both", expand=True)

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Escape>", lambda e: self.root.destroy())

        # 提示文字
        self.canvas.create_text(
            self.root.winfo_screenwidth()//2, 
            self.root.winfo_screenheight()//2, 
            text="请按住鼠标左键画框选择区域 (ESC取消)", 
            fill="white", font=("Arial", 20)
        )

        self.root.mainloop()
        return self.selection

    def on_press(self, event):
        self.start_x = self.root.winfo_pointerx() - self.root.winfo_rootx()
        self.start_y = self.root.winfo_pointery() - self.root.winfo_rooty()
        # Create initial rect
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red', width=3)

    def on_drag(self, event):
        cur_x = self.root.winfo_pointerx() - self.root.winfo_rootx()
        cur_y = self.root.winfo_pointery() - self.root.winfo_rooty()
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_release(self, event):
        end_x = self.root.winfo_pointerx() - self.root.winfo_rootx()
        end_y = self.root.winfo_pointery() - self.root.winfo_rooty()
        
        # Normalize coords
        x1 = min(self.start_x, end_x)
        y1 = min(self.start_y, end_y)
        x2 = max(self.start_x, end_x)
        y2 = max(self.start_y, end_y)
        
        # Ensure valid area
        if x2 - x1 > 10 and y2 - y1 > 10:
            self.selection = f"{x1}, {y1}, {x2}, {y2}"
        
        self.root.destroy()

def open_selector():
    selector = ScreenSelector()
    res = selector.select_area()
    return res or ""  # Return empty string if cancelled

def decode_base64_to_img(base64_str):
    if not base64_str: return None
    img_bytes = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.cvtColor(cv2.imdecode(nparr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

def draw_boxes_local(image, results):
    """
    在本地绘制检测框
    image: RGB numpy array
    results: list of dicts from backend
    """
    if image is None: return None
    
    # Copy to avoid modifying original if needed
    img_draw = image.copy()
    
    for r in results:
        box = r.get('box')
        cls_name = r.get('class_name')
        conf = r.get('conf')
        # Simple hash for color if class_id missing
        cls_id = r.get('class_id', hash(cls_name) % 100)
        
        if box:
            x1, y1, x2, y2 = map(int, box)
            color = random_color(cls_id)
            
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
            
            label = f"{cls_name} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, 0, 0.6, 1)
            cv2.rectangle(img_draw, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(img_draw, label, (x1, y1 - 5), 0, 0.6, (255, 255, 255), 1)
            
    return img_draw

# --- 1. 单图预测 Pipeline ---
def predict_single(image, conf_thres, model_name):
    """
    单图预测 (本地极速版 + 强制640p):
    1. 强制缩小到长边 640
    2. 保存临时文件
    3. 传路径给后端 (Local Handoff)
    4. 本地画图
    """
    if image is None: return None, [["错误", "请上传图片", ""]]
    
    temp_path = None
    scale = 1.0
    
    try:
        # Step 0: 强制 Resize 到 640 (用户需求)
        h, w = image.shape[:2]
        target_size = 640
        
        # 只要有一边超过640，或者为了统一样式，都resize?
        # 通常是缩放到长边为640
        if max(h, w) > target_size: 
            scale = target_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image_processed = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            # 如果小于640，是否要放大？一般不建议放大
            # 如果用户坚持 "改为640"，这里只做缩小处理，保证速度
            image_processed = image

        # Step 1: 保存到本地临时文件
        filename = f"{uuid.uuid4()}.jpg"
        temp_path = os.path.join(TEMP_DIR, filename)
        
        img_bgr = cv2.cvtColor(image_processed, cv2.COLOR_RGB2BGR)
        cv2.imwrite(temp_path, img_bgr)

        # Step 2: 发送请求
        data = {
            'conf': conf_thres, 
            'model_name': model_name, 
            'return_image': 'False',
            'image_path': temp_path 
        }
        
        resp = global_session.post(API_URL, data=data, timeout=10)
        
        if resp.status_code != 200: return image, [["API 错误", resp.text, ""]]
        
        res = resp.json()
        results = res.get('results', [])
        
        # Step 3: 坐标还原 (还原回原图尺寸画框，或者就在640图上画？)
        # 既然用户要看结果，一般希望在上传的原图上画，或者显示的图就是640的
        # 这里我们返回原图(原尺寸)，所以要还原坐标
        if scale != 1.0:
            for r in results:
                if 'box' in r:
                    r['box'] = [c / scale for c in r['box']]

        # Step 4: 本地绘图
        out_img = draw_boxes_local(image, results)
        
        # 转换为表格数据
        df_data = []
        if not results:
             df_data = [["无目标", "-", "-"]]
        else:
            for r in results:
                name = r.get('class_name', 'unknown')
                conf = f"{r.get('conf', 0):.2f}"
                box = r.get('box', [])
                df_data.append([name, conf, str(box)])

        return out_img, df_data
    except Exception as e:
        return image, [["系统错误", str(e), ""]]
    finally:
        if temp_path and os.path.exists(temp_path):
            try: os.remove(temp_path)
            except: pass

# --- 2. 批量预测 Pipeline ---
def predict_batch_pipeline(file_objs, conf_thres, model_name):
    """
    本地极速批量预测: 直接发送文件路径列表给后端 (Local Handoff)
    注: 批量处理不进行图片 Resize (因为不能修改原文件)，依赖后端处理
    """
    if not file_objs: return [], [["无文件", "-", "-"]], None
    
    # Send paths directly
    data = {
        'conf': conf_thres, 
        'model_name': model_name, 
        'return_image': 'False',
        'image_paths': file_objs 
    }
    
    try:
        resp = global_session.post(BATCH_URL, data=data, timeout=60)
 
    except Exception as e:
         return [], [["请求错误", str(e), "-"]], None

    if resp.status_code != 200: return [], [["Backend Error", resp.text, "-"]], None
    
    try:
        result = resp.json()
        batch_results = result.get("batch_results", [])
        
        gallery_images = []
        summary_data = []
        full_results_state = [] 
        
        for idx, item in enumerate(batch_results):
            fname = item.get('filename', 'unknown')
            processed = None
            if idx < len(file_objs):
                fpath = file_objs[idx]
                if os.path.exists(fpath):
                    img_array = np.fromfile(fpath, np.uint8)
                    img_orig = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if img_orig is not None:
                        img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
                        res_list = item.get('results', [])
                        processed = draw_boxes_local(img_rgb, res_list)

            if processed is not None:
                label = f"{fname} ({item['count']})"
                gallery_images.append((processed, label))
                
                count = item.get('count', 0)
                res_list = item.get('results', [])
                stats = {}
                for r in res_list:
                    cname = r.get('class_name', 'unknown')
                    stats[cname] = stats.get(cname, 0) + 1
                
                detail_str = ", ".join([f"{k}:{v}" for k,v in stats.items()]) if stats else "无目标"
                summary_data.append([fname, count, detail_str])
                
                full_results_state.append({
                    "image": processed,
                    "filename": fname,
                    "results": res_list
                })
            
        return gallery_images, summary_data, full_results_state
        
    except Exception as e:
        return [], [["系统错误", str(e), "-"]], None
        
# --- 交互事件处理 ---
def on_select_gallery(evt: gr.SelectData, state):
    # evt.index 是 gallery 中被选中的索引
    if not state or evt.index >= len(state): return None, [["无数据", "-", "-"]]
    
    selected = state[evt.index]
    img = selected['image']
    results = selected['results']
    
    # 构建详细表格
    df_data = []
    if not results:
         df_data = [["无目标", "-", "-"]]
    else:
        for r in results:
            name = r.get('class_name', 'unknown')
            conf = f"{r.get('conf', 0):.2f}"
            box = r.get('box', [])
            box_str = f"[{int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])}]"
            df_data.append([name, conf, box_str])
            
    return img, df_data

def on_select_dataframe(evt: gr.SelectData, state):
    # evt.index[0] 是 dataframe 行索引 (对于 Dataframe, index 是 (row, col))
    row_idx = evt.index[0]
    return on_select_gallery(type('obj', (object,), {'index': row_idx}), state)

    # finally:
    #     for _, f in uploaded_files: f.close()

def random_color(id):
    import colorsys
    h = (((id << 2) ^ 0x937151) % 100) / 100.0
    s = (((id << 3) ^ 0x315793) % 100) / 100.0
    r, g, b = colorsys.hsv_to_rgb(h, s, 1)
    return int(r * 255), int(g * 255), int(b * 255)

# --- 3. 屏幕实时流 Pipeline ---
def predict_screen_stream(conf_thres, model_name, roi_str):
    """
    Generator function that captures screen and yields predicted frames
    Using Optimized Strategy: Downscale Request -> Local Draw
    """
    bbox = None
    if roi_str and "," in roi_str:
        try:
            # Parse "x1,y1,x2,y2"
            parts = list(map(int, roi_str.replace(" ", "").split(',')))
            if len(parts) == 4:
                bbox = tuple(parts)
        except:
            print("Invalid ROI format, using full screen")

    # Reuse session for speed
    session = requests.Session()
    
    while True:
        try:
            # 1. Capture Screen
            screen = ImageGrab.grab(bbox=bbox) 
            img_orig = np.array(screen) # RGB
            h_orig, w_orig = img_orig.shape[:2]
            
            # 2. Downscale for faster upload/inference
            # 640 is typical YOLO size, no need to send 4K screen
            scale_size = 640
            scale = scale_size / max(h_orig, w_orig)
            w_new, h_new = int(w_orig * scale), int(h_orig * scale)
            img_resized = cv2.resize(img_orig, (w_new, h_new))

            # 3. Encode (Faster JPEG)
            img_bgr_small = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
            # JPEG Quality 60 is enough for detection
            _, img_encoded = cv2.imencode('.jpg', img_bgr_small, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            
            files = {'file': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}
            data = {'conf': conf_thres, 'model_name': model_name, 'return_image': 'False'}
            
            # 4. Request (No return image)
            resp = session.post(API_URL, files=files, data=data, timeout=1)
            
            if resp.status_code == 200:
                res = resp.json()
                results = res.get('results', [])
                
                # 5. Draw on Original High-Res Image (Local CPU)
                # Map coordinates back: x_orig = x_pred / scale
                for r in results:
                    box = r.get('box')
                    cls_name = r.get('class_name')
                    conf = r.get('conf')
                    cls_id = r.get('class_id', 0)
                    
                    if box:
                        # Rescale box back to original screen size
                        x1 = int(box[0] / scale)
                        y1 = int(box[1] / scale)
                        x2 = int(box[2] / scale)
                        y2 = int(box[3] / scale)
                        
                        color = random_color(cls_id)
                        
                        # Draw Rect
                        cv2.rectangle(img_orig, (x1, y1), (x2, y2), color, 3)
                        
                        # Draw Label
                        label = f"{cls_name} {conf:.2f}"
                        (w, h), _ = cv2.getTextSize(label, 0, 0.6, 1)
                        cv2.rectangle(img_orig, (x1, y1 - 20), (x1 + w, y1), color, -1)
                        cv2.putText(img_orig, label, (x1, y1 - 5), 0, 0.6, (255, 255, 255), 1)
            
            yield img_orig
            
        except Exception as e:
            # print(f"Screen stream error: {e}")
            yield img_orig # Return raw screen if fail

# --- 4. 获取报表 ---

# --- 4. 获取报表 ---
# 全局缓存变量，避免重复请求静态资源
REPORT_CACHE = {}

def get_report_image(model_dropdown,report_type):
    # 1. 优先检查缓存
    cache_key = f"{model_dropdown}_{report_type}"
    if cache_key in REPORT_CACHE:
        return REPORT_CACHE[cache_key], "获取成功 (来自缓存)"

    # report_type: loss, confusion_matrix, pr_curve
    url = f"{REPORT_URL}/{report_type}"
    params = {"model_name": model_dropdown} if model_dropdown else {}
    
    try:
        # 2. 使用 global_session 复用连接
        resp = global_session.get(url, params=params, timeout=5)
        
        if resp.status_code == 200:
            # Response is raw image bytes
            nparr = np.frombuffer(resp.content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 3. 写入缓存
            REPORT_CACHE[cache_key] = img_rgb
            
            return img_rgb, "获取成功 (服务端)"
        else:
            return None, f"获取失败: {resp.json().get('detail')}"
    except Exception as e:
        return None, str(e)

# --- 构建 UI ---
with gr.Blocks(title="电力巡检系统 v2.0", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚡ 电力巡检图像智能检测系统")
    
    # 全局设置区
    with gr.Row(variant="panel"):
        model_choices = get_available_models()
        default_model = model_choices[0] if model_choices else "yolo11n.pt"
        model_dropdown = gr.Dropdown(model_choices, value=default_model, label="选择模型", scale=2, allow_custom_value=True)
        conf_slider = gr.Slider(0.0, 1.0, value=0.4, label="置信度阈值", scale=2)
        refresh_btn = gr.Button("🔄", scale=0, min_width=50)
        
        def refresh_models():
            ch = get_available_models()
            return gr.Dropdown(choices=ch, value=ch[0] if ch else "")
        refresh_btn.click(refresh_models, outputs=[model_dropdown])

    with gr.Tabs():
        # --- Tab 1: 单图检测 ---
        with gr.Tab("📷 单图检测"):
            with gr.Row():
                with gr.Column():
                    t1_input = gr.Image(type="numpy", label="Input")
                    t1_btn = gr.Button("开始检测", variant="primary")
                with gr.Column():
                    t1_output = gr.Image(type="numpy", label="检测结果")
                    # t1_json = gr.JSON(label="详细数据")
                    t1_table = gr.Dataframe(
                        headers=["类别", "置信度", "坐标位置"],
                        datatype=["str", "str", "str"],
                        label="检测详情",
                        interactive=False
                    )
            
            t1_btn.click(predict_single, [t1_input, conf_slider, model_dropdown], [t1_output, t1_table])

        # --- Tab 2: 批量检测 ---
        with gr.Tab("📂 批量检测"):
            gr.Markdown("支持批量上传多张图片进行处理。点击处理结果图片与表格行可进行联动查看。")
            with gr.Row():
                with gr.Column(scale=1):
                    # file_count="multiple" returns list of file paths
                    t2_input = gr.File(file_count="multiple", type="filepath", label="选择多张图片")
                    t2_btn = gr.Button("批量处理", variant="primary")
                with gr.Column(scale=2):
                    # 总览区域
                    with gr.Group():
                        gr.Markdown("### 1. 结果概览 (Gallery & Summary)")
                        with gr.Row():
                            t2_gallery = gr.Gallery(label="所有图片", columns=4, height=300, allow_preview=False)
                            t2_table_sum = gr.Dataframe(
                                headers=["文件名", "目标数", "统计"],
                                datatype=["str", "number", "str"],
                                label="统计报告 (点击行查看)",
                                interactive=False
                            )

                    # 详情区域
                    gr.Markdown("### 2. 选中详情 (Selected Detail)")
                    with gr.Row():
                        t2_selected_img = gr.Image(label="当前选中图片", type="numpy", height=500)
                        t2_table_detail = gr.Dataframe(
                            headers=["类别", "置信度", "坐标"],
                            datatype=["str", "str", "str"],
                            label="当前图片检测数据",
                            interactive=False
                        )
            
            # State
            t2_state = gr.State()

            # Events
            t2_btn.click(predict_batch_pipeline, 
                         [t2_input, conf_slider, model_dropdown], 
                         [t2_gallery, t2_table_sum, t2_state])
            
            # Linkage
            t2_gallery.select(on_select_gallery, inputs=[t2_state], outputs=[t2_selected_img, t2_table_detail])
            t2_table_sum.select(on_select_dataframe, inputs=[t2_state], outputs=[t2_selected_img, t2_table_detail])

        # --- Tab 3: 屏幕实时检测 ---
        with gr.Tab("🖥️ 屏幕实时检测"):
            gr.Markdown("实时捕获电脑屏幕进行检测 (Screen Capture)")
            
            with gr.Row():
                with gr.Column(scale=4):
                    roi_input = gr.Textbox(
                        label="捕获区域 (x1, y1, x2, y2)", 
                        placeholder="例如: 100, 100, 800, 600 (留空则全屏)",
                        info="请输入坐标或使用右侧按钮选取"
                    )
                with gr.Column(scale=1):
                    select_btn = gr.Button("✂️ 框选区域", min_width=80)
            
            # Selector Action
            select_btn.click(open_selector, outputs=[roi_input])

            with gr.Row():
                with gr.Column(scale=1):
                    start_btn = gr.Button("▶️ 开始屏幕捕获", variant="primary")
                    stop_btn = gr.Button("⏹️ 停止捕获")
                with gr.Column(scale=3):
                    stream_output = gr.Image(label="屏幕检测流", interactive=False)
            
            # Event: Click start to trigger generator, Click stop to cancel
            stream_event = start_btn.click(
                predict_screen_stream, 
                [conf_slider, model_dropdown, roi_input], 
                [stream_output]
            )
            stop_btn.click(fn=None, cancels=[stream_event])

        # --- Tab 4: 模型评估 ---
        with gr.Tab("📊 模型评估"):
            gr.Markdown("查看当前模型在训练集上的表现")
            with gr.Row():
                btn_cm = gr.Button("混淆矩阵 (Confusion Matrix)")
                btn_pr = gr.Button("PR 曲线")
                btn_loss = gr.Button("训练 Loss")
            
            with gr.Row():
                report_img = gr.Image(label="评估图表")
                report_msg = gr.Textbox(label="状态", interactive=False)
            
            btn_cm.click(lambda m: get_report_image(m, "confusion_matrix"), inputs=[model_dropdown], outputs=[report_img, report_msg])
            btn_pr.click(lambda m: get_report_image(m, "pr_curve"), inputs=[model_dropdown], outputs=[report_img, report_msg])
            btn_loss.click(lambda m: get_report_image(m, "loss"), inputs=[model_dropdown], outputs=[report_img, report_msg])
            
        # --- Tab 5: 系统说明 ---
        with gr.Tab("ℹ️ 关于系统"):
            gr.Markdown("""
            ## 🎓 课设项目演示系统
            
            本系统集成了两个关键技术模块：
            
            1.  **高精度缺陷检测模型 (Task 1)**
                *   模型架构: YOLOv11s
                *   框架: PyTorch 1.12+ (Training), ONNX (Intermediate)
                *   数据集: 绝缘子缺陷数据集 (VOC/YOLO格式)
                *   增强: Mosaic, Mixup, HSV Augmentation
            
            2.  **高性能推理加速 (Task 2)**
                *   推理引擎: NVIDIA TensorRT 8.x
                *   精度优化: FP16 / INT8 Quantization
                *   服务化: C++ Inference Service (Backend) + FastAPI
            
            **开发栈:**
            *   Frontend: Gradio / Python
            *   Backend: FastAPI / C++ / TensorRT
            *   CV Lib: OpenCV 4.x
            """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)