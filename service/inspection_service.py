import os
import glob
from service.detector import Detector
from service.picture_processing import picture

class InspectionService:
    def __init__(self, model_path=None):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.yolo_root = os.path.join(self.project_root, "yolov11")
        
        if model_path is None:
             # Default path relative to project root
             model_path = os.path.join(self.yolo_root, "yolo11n.pt")
        
        self.current_model_path = model_path
        self.detector = Detector(model_path)
        self.processor = picture()

    def get_available_models(self):
        """
        Scan for .pt files in yolov11/ and yolov11/runs/detect/*/weights/
        Returns a list of model names (relative paths or filenames convenient for selection)
        """
        models = []
        
        # 1. Base models in yolov11/
        base_models = glob.glob(os.path.join(self.yolo_root, "*.pt"))
        for m in base_models:
            models.append({
                "name": os.path.basename(m),
                "path": m,
                "type": "base"
            })
            
        # 2. Trained models in yolov11/runs/detect/*/weights/*.pt
        # Pattern: yolov11/runs/detect/**/weights/*.pt
        runs_pattern = os.path.join(self.yolo_root, "runs", "detect", "**", "weights", "*.pt")
        trained_models = glob.glob(runs_pattern, recursive=True)
        
        for m in trained_models:
            # Construct a readable name, e.g., "train/best.pt"
            # Relpath from 'runs/detect' to give context like "train2\weights\best.pt"
            try:
                rel_start = os.path.join(self.yolo_root, "runs", "detect")
                rel_path = os.path.relpath(m, rel_start)
                display_name = f"Run: {rel_path}"
            except ValueError:
                display_name = os.path.basename(m)

            models.append({
                "name": display_name,
                "path": m,
                "type": "trained"
            })
            
        return models

    def reload_model(self, model_name):
        """
        Reload detector with a specific model if it's different from current.
        """
        if not model_name:
            return False

        available_models = self.get_available_models()
        target_path = None
        
        # Try to match by name
        for m in available_models:
            if m["name"] == model_name:
                target_path = m["path"]
                break
        
        # If not matched by name, maybe it's the raw path
        if not target_path:
             for m in available_models:
                if m["path"] == model_name:
                    target_path = m["path"]
                    break

        if target_path and self.current_model_path != target_path:
            # print(f"Switching model from {self.current_model_path} to {target_path}")
            try:
                self.detector = Detector(target_path)
                self.current_model_path = target_path
                return True
            except Exception as e:
                print(f"Failed to load model {target_path}: {e}")
                return False
                
        return False

    def process_request(self, file_contents, conf=0.4, model_name=None, return_image=True):
        """
        Process an image upload request: decode, detect, draw.
        Returns: dict with 'results' (list) and 'processed_image' (numpy array)
        """
        # 0. Switch Model if requested
        if model_name:
            self.reload_model(model_name)

        # 1. Decode
        img = self.processor.decode_image_from_bytes(file_contents)
        if img is None:
            raise ValueError("Invalid image format")

        # 2. Predict
        results = self.detector.predict(img, conf_thres=conf)

        # 3. Generate Image (Draw boxes)
        processed_base64 = None
        if return_image:
            processed_img = self.processor.draw_boxes(img, results)
            processed_base64 = self.processor.encode_image_to_base64(processed_img)

        return {
            "results": results,
            # "processed_image": processed_img, # Optimize memory
            "processed_base64": processed_base64,
            "model_path": self.current_model_path
        }
