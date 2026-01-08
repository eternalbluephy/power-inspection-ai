import multiprocessing as mp
import platform

from ultralytics import YOLO


def main():
    model = YOLO("yolo11n.pt", task="detect")
    results = model.train(
        data="data.yaml",
        epochs=100,
        batch=32,
        imgsz=640,
        device=0,
        workers=12,
        seed=42,
        pretrained=True,
        cache=True,
        verbose=True,
    )
    print(results)


if __name__ == "__main__":
    if platform.system() == "Windows":
        mp.freeze_support()
    main()
