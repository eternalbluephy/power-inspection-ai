import numpy as np
import torch

_INPUT_HW = (1280, 1280)
_STRIDES = (8, 16, 32)
_FEAT_SHAPES = ((160, 160), (80, 80), (40, 40))
_REG_MAX = 16
_NUM_PRED = 33600
_TOPK_PRE_NMS = 1000


def _build_anchors_and_strides():
    anchors = []
    strides = []

    for (h, w), s in zip(_FEAT_SHAPES, _STRIDES):
        xs = torch.arange(w, dtype=torch.float32) + 0.5
        ys = torch.arange(h, dtype=torch.float32) + 0.5
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        anchor = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
        anchors.append(anchor)
        strides.append(torch.full((h * w, 1), float(s), dtype=torch.float32))

    anchors = torch.cat(anchors, dim=0)
    strides = torch.cat(strides, dim=0)
    return anchors, strides


_ANCHORS_CPU, _STRIDES_CPU = _build_anchors_and_strides()
_DFL_VAL_CPU = torch.arange(_REG_MAX, dtype=torch.float32)

_DEVICE_CACHE: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _get_cached_tensors(device: torch.device):
    if device.type == "cpu":
        return _ANCHORS_CPU, _STRIDES_CPU, _DFL_VAL_CPU

    key = str(device)
    cached = _DEVICE_CACHE.get(key)
    if cached is not None:
        return cached

    anchors = _ANCHORS_CPU.to(device)
    strides = _STRIDES_CPU.to(device)
    dfl_val = _DFL_VAL_CPU.to(device)
    _DEVICE_CACHE[key] = (anchors, strides, dfl_val)
    return anchors, strides, dfl_val


def get_anchors():
    return _ANCHORS_CPU, _STRIDES_CPU


def nms_python(boxes, scores, iou_threshold):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i.item())
        if order.numel() == 1:
            break

        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])

        w = (xx2 - xx1).clamp(min=0.0)
        h = (yy2 - yy1).clamp(min=0.0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = torch.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return torch.tensor(keep, dtype=torch.long)


def decode_outputs(output, conf_thres=0.25, iou_thres=0.45):
    with torch.no_grad():
        if isinstance(output, np.ndarray):
            output = torch.from_numpy(output)

        if (
            output.ndim != 3
            or output.shape[0] != 1
            or output.shape[1] != _NUM_PRED
            or output.shape[2] != 68
        ):
            return torch.empty(0), torch.empty(0), torch.empty(0)

        device = output.device
        anchors, strides, dfl_val = _get_cached_tensors(device)

        cls_logits = output[0, :, 64:]
        scores_all = cls_logits.sigmoid()
        max_scores, classes = torch.max(scores_all, dim=1)
        keep_idx = torch.where(max_scores > conf_thres)[0]
        if keep_idx.numel() == 0:
            return torch.empty(0), torch.empty(0), torch.empty(0)

        if keep_idx.numel() > _TOPK_PRE_NMS:
            kept_scores = max_scores[keep_idx]
            topk_scores, topk_pos = torch.topk(
                kept_scores, k=_TOPK_PRE_NMS, largest=True, sorted=False
            )
            keep_idx = keep_idx[topk_pos]
            scores = topk_scores
            classes = classes[keep_idx]
        else:
            scores = max_scores[keep_idx]
            classes = classes[keep_idx]

        box_raw = (
            output[0, keep_idx, :64].contiguous().view(-1, 4, _REG_MAX)
        )
        box_prob = box_raw.softmax(dim=2)
        dist = (box_prob * dfl_val.view(1, 1, _REG_MAX)).sum(dim=2)

        anchors_kept = anchors[keep_idx]
        strides_kept = strides[keep_idx]

        x1 = anchors_kept[:, 0] - dist[:, 0]
        y1 = anchors_kept[:, 1] - dist[:, 1]
        x2 = anchors_kept[:, 0] + dist[:, 2]
        y2 = anchors_kept[:, 1] + dist[:, 3]
        bboxes = torch.stack([x1, y1, x2, y2], dim=1) * strides_kept

        try:
            import torchvision

            if hasattr(torchvision.ops, "batched_nms"):
                indices = torchvision.ops.batched_nms(
                    bboxes, scores, classes, iou_thres
                )
            else:
                indices = torchvision.ops.nms(bboxes, scores, iou_thres)
        except Exception:
            indices = nms_python(bboxes, scores, iou_thres)

        return bboxes[indices], scores[indices], classes[indices]
