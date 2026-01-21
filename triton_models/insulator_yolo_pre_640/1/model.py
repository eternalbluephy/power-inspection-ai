import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        self.inp_h = 640
        self.inp_w = 640
        self.pad_val = 114.0 / 255.0

    def execute(self, requests):
        resps = []
        for req in requests:
            jpeg = pb_utils.get_input_tensor_by_name(req, "jpeg").as_numpy().reshape(-1)[0]
            buf = np.frombuffer(jpeg, dtype=np.uint8)
            t = torch.from_numpy(buf)
            img = torchvision.io.decode_jpeg(t)          # uint8, (C,H,W), RGB
            img = img.float() / 255.0                    # FP32

            _, h, w = img.shape
            scale = min(self.inp_w / w, self.inp_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            pad_x = (self.inp_w - new_w) // 2
            pad_y = (self.inp_h - new_h) // 2

            x = img.unsqueeze(0)                         # (1,3,H,W)
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)

            out = torch.full((1, 3, self.inp_h, self.inp_w), self.pad_val, dtype=torch.float32)
            out[:, :, pad_y:pad_y+new_h, pad_x:pad_x+new_w] = x

            images = out.numpy()
            meta = np.array([[scale, pad_x, pad_y, w, h]], dtype=np.float32)

            resps.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("images", images),
                pb_utils.Tensor("meta", meta),
            ]))
        return resps
