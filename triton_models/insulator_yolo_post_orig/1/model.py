import json

import numpy as np
import torch
import torchvision
import triton_python_backend_utils as pb_utils

_NUM_PRED = 33600
_REG_MAX = 16


class TritonPythonModel:
    def initialize(self, args):
        cfg = json.loads(args["model_config"])
        p = cfg.get("parameters", {})
        self.conf = float(p.get("conf_thres", {}).get("string_value", "0.25"))
        self.iou = float(p.get("iou_thres", {}).get("string_value", "0.45"))
        self.max_det = int(p.get("max_det", {}).get("string_value", "300"))
        self.topk = int(p.get("topk", {}).get("string_value", "1000"))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dfl_val = torch.arange(_REG_MAX, dtype=torch.float32, device=self.device)

        self.anchors, self.strides = self._make_anchors_1280(self.device)

    def _make_anchors_1280(self, device):
        strides = [8, 16, 32]
        feats = [1280 // s for s in strides]

        all_xy = []
        all_s = []
        for s, f in zip(strides, feats):
            y, x = torch.meshgrid(
                torch.arange(f, device=device),
                torch.arange(f, device=device),
                indexing="ij",
            )
            xy = torch.stack((x + 0.5, y + 0.5), dim=-1).view(-1, 2)
            all_xy.append(xy)
            all_s.append(torch.full((xy.shape[0], 1), float(s), device=device))

        anchors = torch.cat(all_xy, dim=0)
        strides = torch.cat(all_s, dim=0)
        assert anchors.shape[0] == _NUM_PRED
        return anchors, strides

    def execute(self, requests):
        resps = []
        for req in requests:
            out = pb_utils.get_input_tensor_by_name(req, "output").as_numpy()  # (1,33600,68)
            output = torch.from_numpy(out).to(self.device, non_blocking=False)

            cls_logits = output[0, :, 64:]  # (33600,4)
            scores_all = cls_logits.sigmoid()
            max_scores, classes = torch.max(scores_all, dim=1)
            keep = torch.where(max_scores > self.conf)[0]

            if keep.numel() == 0:
                resps.append(self._empty())
                continue

            if keep.numel() > self.topk:
                kept_scores = max_scores[keep]
                topk_scores, topk_pos = torch.topk(
                    kept_scores, k=self.topk, largest=True, sorted=False
                )
                keep = keep[topk_pos]
                scores = topk_scores
                classes = classes[keep]
            else:
                scores = max_scores[keep]
                classes = classes[keep]

            box_raw = output[0, keep, :64].contiguous().view(-1, 4, _REG_MAX)
            box_prob = box_raw.softmax(dim=2)
            dist = (box_prob * self.dfl_val.view(1, 1, _REG_MAX)).sum(dim=2)  # (N,4)

            a = self.anchors[keep]  # (N,2)
            s = self.strides[keep]  # (N,1)

            x1 = a[:, 0] - dist[:, 0]
            y1 = a[:, 1] - dist[:, 1]
            x2 = a[:, 0] + dist[:, 2]
            y2 = a[:, 1] + dist[:, 3]
            boxes = torch.stack([x1, y1, x2, y2], dim=1) * s  # (N,4) 1280-letterbox 像素坐标

            idx = torchvision.ops.batched_nms(boxes, scores, classes, self.iou)
            boxes = boxes[idx][: self.max_det]
            scores = scores[idx][: self.max_det]
            classes = classes[idx][: self.max_det]

            # meta: [scale, pad_x, pad_y, orig_w, orig_h]
            meta = pb_utils.get_input_tensor_by_name(req, "meta").as_numpy()
            m = torch.from_numpy(meta).to(self.device, non_blocking=False)
            scale = m[0, 0]
            pad_x = m[0, 1]
            pad_y = m[0, 2]
            orig_w = m[0, 3]
            orig_h = m[0, 4]

            boxes[:, 0] = (boxes[:, 0] - pad_x) / scale
            boxes[:, 1] = (boxes[:, 1] - pad_y) / scale
            boxes[:, 2] = (boxes[:, 2] - pad_x) / scale
            boxes[:, 3] = (boxes[:, 3] - pad_y) / scale
            boxes[:, 0].clamp_(0, orig_w)
            boxes[:, 2].clamp_(0, orig_w)
            boxes[:, 1].clamp_(0, orig_h)
            boxes[:, 3].clamp_(0, orig_h)

            num = np.array([boxes.shape[0]], dtype=np.int32)

            boxes_np = np.zeros((self.max_det, 4), np.float32)
            scores_np = np.zeros((self.max_det,), np.float32)
            classes_np = np.zeros((self.max_det,), np.int32)

            n = boxes.shape[0]
            if n > 0:
                boxes_np[:n] = boxes.detach().to("cpu").numpy().astype(np.float32)
                scores_np[:n] = scores.detach().to("cpu").numpy().astype(np.float32)
                classes_np[:n] = classes.detach().to("cpu").numpy().astype(np.int32)

            resps.append(self._resp(num, boxes_np, scores_np, classes_np))
        return resps

    def _empty(self):
        num = np.array([0], dtype=np.int32)
        return self._resp(
            num,
            np.zeros((self.max_det, 4), np.float32),
            np.zeros((self.max_det,), np.float32),
            np.zeros((self.max_det,), np.int32),
        )

    def _resp(self, num, boxes, scores, classes):
        return pb_utils.InferenceResponse(
            output_tensors=[
                pb_utils.Tensor("num_dets", num),
                pb_utils.Tensor("boxes", boxes),
                pb_utils.Tensor("scores", scores),
                pb_utils.Tensor("classes", classes),
            ]
        )
