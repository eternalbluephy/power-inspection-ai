import cv2
import numpy as np
USE_TENSORRT=False
class Detector:
    def __init__(self,model_path):
        if USE_TENSORRT:
            print("Using TensorRT")
            pass
        else:
            print("Using PyTorch")
            from ultralytics import YOLO
            try:
                self.model= YOLO(model_path)
                print(f"Using {model_path}")
            except:
                print(f"{model_path} is found failed ,using base")
                self.model=YOLO("yolo11n.pt")
    def predict(self, image, conf_thres=0.25):
        if isinstance(image,str):
            img=cv2.imread(image)
        else :
            img=image
        if img is None:
            raise ValueError("image is Empty")
        results=self.model(img, conf=conf_thres, verbose=False)[0]
        output_data=[]
        boxes =results.boxes.data.tolist()
        print(f"finding {len(boxes)} data")
        for box in boxes:
            x1,y1,x2,y2=map(int,box[:4])
            conf=float(box[4])
            cls_id=int(box[5])
            obj_info={
                "box":[x1,y1,x2,y2],
                "conf":conf,
                "class_id": cls_id,
                "class_name": results.names[cls_id]
            }
            output_data.append(obj_info)
        return output_data
        


