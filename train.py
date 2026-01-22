import multiprocessing as mp
import platform

from ultralytics import YOLO


def main():
    model = YOLO("yolo11n.pt", task="detect")
    results = model.train(
        data="data.yaml",
        epochs=150,
        batch=64,
        imgsz=640,
        device=0,
        workers=4,
        seed=42,
        pretrained=True,
        cache=False,
        verbose=True,
        hsv_h=0.015,
        mosaic=1.0,
        close_mosaic=20,
    )
    print(results)


if __name__ == "__main__":
    if platform.system() == "Windows":
        mp.freeze_support()
    main()
