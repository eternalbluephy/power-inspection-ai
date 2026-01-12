import cv2
import numpy as np

class picture:
    def hsv2bgr(self,h, s, v):
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

    def random_color(self,id):
        h_plane = (((id << 2) ^ 0x937151) % 100) / 100.0
        s_plane = (((id << 3) ^ 0x315793) % 100) / 100.0
        return self.hsv2bgr(h_plane, s_plane, 1)

    def read_img(self,img_path):
        img=cv2.imread(img_path)
        if img is None:
            return "错误: 无法读取图片，请检查路径。"
        else :
            return img

    def decode_image_from_bytes(self, content_bytes):
        """Decode bytes to OpenCV image"""
        nparr = np.frombuffer(content_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    
    def encode_image_to_base64(self, img):
        """Encode OpenCV image to base64 string"""
        _, buffer = cv2.imencode('.jpg', img)
        import base64
        return base64.b64encode(buffer).decode('utf-8')

    def draw_boxes(self, img, results):
        """Draw bounding boxes on the image with transparency"""
        if img is None:
            return None
            
        img_copy = img.copy()
        overlay = img.copy() # 用于绘制半透明图层
        
        for item in results:
            box = item['box']
            conf = item['conf']
            cls_id = item['class_id']
            cls_name = item['class_name']
            
            x1, y1, x2, y2 = map(int, box)
            color = self.random_color(cls_id)
            
            # 1. 绘制半透明填充矩形
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            
            # 2. 绘制实线边框
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{cls_name} {conf:.2f}"
            t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img_copy, (x1, y1), c2, color, -1, cv2.LINE_AA)
            cv2.putText(img_copy, label, (x1, y1 - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
            
        # 3. 混合图层实现透明效果 (alpha=0.3)
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, img_copy, 1 - alpha, 0, img_copy)
            
        return img_copy