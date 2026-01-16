import os
import time
from typing import Any

import cv2
import numpy as np
import tritonclient.grpc as grpcclient
from fastapi import FastAPI, File, Query, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TRITON_URL = os.getenv("TRITON_GRPC_URL", "localhost:8001")
try:
    triton_client = grpcclient.InferenceServerClient(url=TRITON_URL)
    print(f"Connected to Triton at {TRITON_URL}")
except Exception as e:
    print(f"Failed to connect to Triton: {e}")
    triton_client = None

LABELS = ["ring_shifted", "nest", "broken", "burn"]
_MODEL_IO_CACHE: dict[str, tuple[str, str, list[str]]] = {}


def _get_model_io(model_name: str) -> tuple[str, str, list[str]]:
    """
    Returns: (input_name, input_datatype, output_names) from Triton model metadata.
    Cached to avoid per-frame metadata RPCs.
    """
    if model_name in _MODEL_IO_CACHE:
        return _MODEL_IO_CACHE[model_name]
    if not triton_client:
        raise RuntimeError("Triton not connected")

    meta = triton_client.get_model_metadata(model_name=model_name)
    # gRPC metadata object: meta.inputs / meta.outputs
    if not hasattr(meta, "inputs") or not meta.inputs:
        raise RuntimeError(f"Model {model_name} has no inputs in metadata")

    inp0 = meta.inputs[0]
    input_name = inp0.name
    input_datatype = inp0.datatype
    output_names = [o.name for o in getattr(meta, "outputs", [])]

    _MODEL_IO_CACHE[model_name] = (input_name, input_datatype, output_names)
    return input_name, input_datatype, output_names


def _as_1d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    return a.reshape(-1)


def _as_boxes(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    return a.reshape(-1, 4)


def _maybe_denormalize_boxes(
    boxes_xyxy: np.ndarray, input_w: int, input_h: int
) -> np.ndarray:
    if boxes_xyxy.size == 0:
        return boxes_xyxy

    # 若坐标均在 [0,1] 附近，认为是归一化坐标
    max_v = float(np.max(boxes_xyxy))
    if max_v <= 1.5:
        out = boxes_xyxy.copy()
        out[:, 0] *= float(input_w)
        out[:, 2] *= float(input_w)
        out[:, 1] *= float(input_h)
        out[:, 3] *= float(input_h)
        return out
    return boxes_xyxy


def _ensure_xyxy(boxes: np.ndarray) -> np.ndarray:
    """尽量把 boxes 变成 xyxy。若发现大量 x2<x1 或 y2<y1，尝试从 yxyx 转换。"""
    if boxes.size == 0:
        return boxes

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    invalid = np.mean((x2 < x1) | (y2 < y1))
    if invalid > 0.5:
        # yxyx -> xyxy: [y1,x1,y2,x2] => [x1,y1,x2,y2]
        return boxes[:, [1, 0, 3, 2]]
    return boxes


def _parse_triton_ensemble_outputs(
    res: Any,
    input_w: int,
    input_h: int,
    original_w: int,
    original_h: int,
    scale: float,
    pad_x: int,
    pad_y: int,
):
    """解析 Triton ensemble 输出并映射回原图坐标。

    期望输出：
    - num_dets: (1,) INT32
    - boxes: (300,4) FP32  (可能也会是 (1,300,4))
    - scores: (300,) FP32  (可能也会是 (1,300))
    - classes: (300,) INT32 (可能也会是 (1,300))
    """
    num_dets = res.as_numpy("num_dets")
    boxes = res.as_numpy("boxes")
    scores = res.as_numpy("scores")
    classes = res.as_numpy("classes")

    if num_dets is None or boxes is None or scores is None or classes is None:
        raise RuntimeError("Missing one or more outputs: num_dets/boxes/scores/classes")

    n = int(_as_1d(num_dets)[0])
    boxes = _as_boxes(boxes)
    scores = _as_1d(scores)
    classes = _as_1d(classes)

    n = max(0, min(n, boxes.shape[0], scores.shape[0], classes.shape[0]))
    boxes = boxes[:n]
    scores = scores[:n]
    classes = classes[:n]

    # 过滤低置信度并限制 Top-K，避免视频流每帧 JSON 过大导致卡顿
    try:
        score_thres = float(os.getenv("TRITON_SCORE_THRES", "0.25"))
    except Exception:
        score_thres = 0.25
    try:
        topk = int(os.getenv("TRITON_TOPK", "80"))
    except Exception:
        topk = 80

    if scores.size > 0:
        keep = scores >= score_thres
        if np.any(keep):
            boxes = boxes[keep]
            scores = scores[keep]
            classes = classes[keep]
        else:
            boxes = boxes[:0]
            scores = scores[:0]
            classes = classes[:0]

    if topk > 0 and scores.size > topk:
        idx = np.argpartition(-scores, kth=topk - 1)[:topk]
        idx = idx[np.argsort(-scores[idx])]
        boxes = boxes[idx]
        scores = scores[idx]
        classes = classes[idx]

    # 允许通过环境变量强制指定 boxes 顺序（默认自动判断）
    box_order = os.getenv("TRITON_BOX_ORDER", "auto").lower().strip()
    if box_order == "yxyx":
        boxes = boxes[:, [1, 0, 3, 2]]
    elif box_order in ("xyxy", "auto"):
        pass
    else:
        print(f"Unknown TRITON_BOX_ORDER={box_order}, using auto")

    boxes = _maybe_denormalize_boxes(boxes, input_w=input_w, input_h=input_h)
    if box_order == "auto":
        boxes = _ensure_xyxy(boxes)

    detections = []
    for box, score, cls in zip(boxes, scores, classes):
        x1 = (float(box[0]) - float(pad_x)) / float(scale)
        y1 = (float(box[1]) - float(pad_y)) / float(scale)
        x2 = (float(box[2]) - float(pad_x)) / float(scale)
        y2 = (float(box[3]) - float(pad_y)) / float(scale)

        x1 = max(0.0, min(x1, float(original_w)))
        y1 = max(0.0, min(y1, float(original_h)))
        x2 = max(0.0, min(x2, float(original_w)))
        y2 = max(0.0, min(y2, float(original_h)))

        cls_i = int(cls)
        detections.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "score": float(score),
                "label": LABELS[cls_i] if 0 <= cls_i < len(LABELS) else str(cls_i),
            }
        )

    return detections


def _parse_triton_boxes_outputs(
    res: Any,
    original_w: int,
    original_h: int,
):
    """Parse ensemble outputs where boxes are already in original image coordinates."""
    num_dets = res.as_numpy("num_dets")
    boxes = res.as_numpy("boxes")
    scores = res.as_numpy("scores")
    classes = res.as_numpy("classes")

    if num_dets is None or boxes is None or scores is None or classes is None:
        raise RuntimeError("Missing one or more outputs: num_dets/boxes/scores/classes")

    n = int(_as_1d(num_dets)[0])
    boxes = _as_boxes(boxes)
    scores = _as_1d(scores)
    classes = _as_1d(classes)

    n = max(0, min(n, boxes.shape[0], scores.shape[0], classes.shape[0]))
    boxes = boxes[:n]
    scores = scores[:n]
    classes = classes[:n]

    try:
        score_thres = float(os.getenv("TRITON_SCORE_THRES", "0.25"))
    except Exception:
        score_thres = 0.25
    try:
        topk = int(os.getenv("TRITON_TOPK", "80"))
    except Exception:
        topk = 80

    if scores.size > 0:
        keep = scores >= score_thres
        if np.any(keep):
            boxes = boxes[keep]
            scores = scores[keep]
            classes = classes[keep]
        else:
            boxes = boxes[:0]
            scores = scores[:0]
            classes = classes[:0]

    if topk > 0 and scores.size > topk:
        idx = np.argpartition(-scores, kth=topk - 1)[:topk]
        idx = idx[np.argsort(-scores[idx])]
        boxes = boxes[idx]
        scores = scores[idx]
        classes = classes[idx]

    # Handle possible normalized boxes and/or swapped order.
    box_order = os.getenv("TRITON_BOX_ORDER", "auto").lower().strip()
    if box_order == "yxyx":
        boxes = boxes[:, [1, 0, 3, 2]]
    elif box_order in ("xyxy", "auto"):
        pass
    else:
        print(f"Unknown TRITON_BOX_ORDER={box_order}, using auto")

    boxes = _maybe_denormalize_boxes(boxes, input_w=original_w, input_h=original_h)
    if box_order == "auto":
        boxes = _ensure_xyxy(boxes)

    detections = []
    for box, score, cls in zip(boxes, scores, classes):
        x1 = max(0.0, min(float(box[0]), float(original_w)))
        y1 = max(0.0, min(float(box[1]), float(original_h)))
        x2 = max(0.0, min(float(box[2]), float(original_w)))
        y2 = max(0.0, min(float(box[3]), float(original_h)))

        cls_i = int(cls)
        detections.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "score": float(score),
                "label": LABELS[cls_i] if 0 <= cls_i < len(LABELS) else str(cls_i),
            }
        )
    return detections


def letterbox_image(img, new_shape=(1280, 1280), color=(114, 114, 114)):
    original_h, original_w = img.shape[:2]
    target_w, target_h = new_shape
    scale = min(target_w / original_w, target_h / original_h)
    new_w = int(original_w * scale)
    new_h = int(original_h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), color, dtype=np.uint8)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    return canvas, scale, pad_x, pad_y


@app.get("/models")
async def get_models():
    if not triton_client:
        return []
    try:
        # Try to get models from Triton
        resp = triton_client.get_model_repository_index()
        # gRPC: RepositoryIndexResponse(models=[...])
        if hasattr(resp, "models"):
            return [m.name for m in resp.models]

        # Fallback: list[dict] (兼容可能的不同 client 版本)
        if isinstance(resp, list):
            out = []
            for m in resp:
                if isinstance(m, dict) and "name" in m:
                    out.append(m["name"])
                elif hasattr(m, "name"):
                    out.append(getattr(m, "name"))
            return out

        return []
    except Exception as e:
        print(f"Error fetching models: {e}")
        # Fallback for dev/testing
        return ["yolo_trt_model"]


def process_frame(image_bytes, model_name="yolo_trt_model"):
    if not triton_client:
        return [], (0, 0), None
    try:
        input_name, input_datatype, output_names = _get_model_io(model_name)

        t0 = time.perf_counter()
        if str(input_datatype).upper() == "BYTES":
            # Forward JPEG bytes directly to Triton (preprocess moved into Triton ensemble).
            # We still decode locally to get original width/height for UI scaling.
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR
            if img is None:
                return [], (0, 0), None
            t_decode = time.perf_counter()
            original_h, original_w = img.shape[:2]

            inp = np.array([image_bytes], dtype=object)
            inputs = [grpcclient.InferInput(input_name, inp.shape, "BYTES")]
            inputs[0].set_data_from_numpy(inp)

            # Prefer requesting known detection outputs; otherwise request all outputs.
            prefer = ["num_dets", "boxes", "scores", "classes"]
            req = prefer if all(n in output_names for n in prefer) else output_names
            outputs = [grpcclient.InferRequestedOutput(n) for n in req]

            t_before_triton = time.perf_counter()
            res = triton_client.infer(model_name, inputs, outputs=outputs)
            t_after_triton = time.perf_counter()

            detections = _parse_triton_boxes_outputs(
                res, original_w=original_w, original_h=original_h
            )
            t_parse = time.perf_counter()

            timings = {
                "decode_ms": (t_decode - t0) * 1000.0,
                "letterbox_ms": 0.0,
                "blob_ms": 0.0,
                "triton_ms": (t_after_triton - t_before_triton) * 1000.0,
                "post_ms": (t_parse - t_after_triton) * 1000.0,
                "total_ms": (t_parse - t0) * 1000.0,
            }
            return detections, (original_w, original_h), timings

        # Default path: local preprocess to FP32 tensor then infer.
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR
        if img is None:
            return [], (0, 0), None
        t_decode = time.perf_counter()

        original_h, original_w = img.shape[:2]

        img_resized, scale, pad_x, pad_y = letterbox_image(img, (1280, 1280))
        t_letterbox = time.perf_counter()

        img_data = cv2.dnn.blobFromImage(
            img_resized,
            scalefactor=1.0 / 255.0,
            size=(1280, 1280),
            mean=(0.0, 0.0, 0.0),
            swapRB=True,
            crop=False,
        )
        t_blob = time.perf_counter()

        inputs = [grpcclient.InferInput(input_name, img_data.shape, "FP32")]
        inputs[0].set_data_from_numpy(img_data)

        prefer = ["num_dets", "boxes", "scores", "classes"]
        req = prefer if all(n in output_names for n in prefer) else output_names
        outputs = [grpcclient.InferRequestedOutput(n) for n in req]

        t_before_triton = time.perf_counter()
        res = triton_client.infer(model_name, inputs, outputs=outputs)
        t_after_triton = time.perf_counter()
        if all(n in output_names for n in ["num_dets", "boxes", "scores", "classes"]):
            detections = _parse_triton_ensemble_outputs(
                res,
                input_w=1280,
                input_h=1280,
                original_w=original_w,
                original_h=original_h,
                scale=scale,
                pad_x=pad_x,
                pad_y=pad_y,
            )
        else:
            # Raw model output (e.g. "output") is not supported here.
            raise RuntimeError(
                f"Model {model_name} does not expose ensemble outputs; got outputs={output_names}"
            )

        t_parse = time.perf_counter()
        timings = {
            "decode_ms": (t_decode - t0) * 1000.0,
            "letterbox_ms": (t_letterbox - t_decode) * 1000.0,
            "blob_ms": (t_blob - t_letterbox) * 1000.0,
            "triton_ms": (t_after_triton - t_before_triton) * 1000.0,
            "post_ms": (t_parse - t_after_triton) * 1000.0,
            "total_ms": (t_parse - t0) * 1000.0,
        }

        return detections, (original_w, original_h), timings
    except Exception as e:
        print(f"Inference error: {e}")
        return [], (0, 0), None


@app.post("/detect/image")
async def detect_image(
    file: UploadFile = File(...), model_name: str = Query("yolo_trt_model")
):
    if not triton_client:
        return {"error": "Triton not connected", "detections": []}
    contents = await file.read()
    detections, size, timings = await run_in_threadpool(
        process_frame, contents, model_name
    )
    return {
        "detections": detections,
        "image_size": [size[0], size[1]],
        "timing": timings,
    }


@app.websocket("/detect/stream")
async def websocket_endpoint(websocket: WebSocket, model_name: str = "yolo_trt_model"):
    await websocket.accept()
    # model_name is now populated from query param
    try:
        # First message could be config
        # But for simplicity assume stream of blobs
        while True:
            data = await websocket.receive_bytes()

            # Check if it's a config message (text) or image (bytes) is tricky if relying on types
            # But receive_bytes matches binary.

            start = time.time()
            detections, size, timings = process_frame(data, model_name)
            process_time = (time.time() - start) * 1000
            if timings and isinstance(timings, dict) and "total_ms" in timings:
                process_time = float(timings["total_ms"])

            await websocket.send_json(
                {
                    "detections": detections,
                    "image_size": [size[0], size[1]],
                    "inference_time": process_time,
                    "timing": timings,
                }
            )
    except Exception as e:
        print(f"WebSocket error: {e}")
