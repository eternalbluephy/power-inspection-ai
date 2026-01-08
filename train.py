from ultralytics import YOLO

model = YOLO("yolo11n.pt", task="detect")
results = model.train(
    data="data.yaml",
    epochs=100,
    batch=16,
    imgsz=960,
    device=0,
    workers=12,
    seed=42,
    pretrained=True,
    cache=True,
    verbose=True,
)
