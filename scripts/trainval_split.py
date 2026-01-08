import random
import shutil
from pathlib import Path


def create_yolo_split(
    source_images_dir: str,
    source_labels_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42,
):
    random.seed(seed)

    source_images_path = Path(source_images_dir)
    source_labels_path = Path(source_labels_dir)
    output_path = Path(output_dir)

    train_images_dir = output_path / "train" / "images"
    train_labels_dir = output_path / "train" / "labels"
    val_images_dir = output_path / "val" / "images"
    val_labels_dir = output_path / "val" / "labels"

    for directory in [
        train_images_dir,
        train_labels_dir,
        val_images_dir,
        val_labels_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"}
    all_images = [
        f
        for f in source_images_path.iterdir()
        if f.is_file() and f.suffix in image_extensions
    ]

    print(f"找到 {len(all_images)} 张图像")

    random.shuffle(all_images)

    split_index = int(len(all_images) * train_ratio)
    train_images = all_images[:split_index]
    val_images = all_images[split_index:]

    print(f"训练集: {len(train_images)} 张图像")
    print(f"验证集: {len(val_images)} 张图像")

    print("\n正在复制训练集")
    for img_path in train_images:
        shutil.copy2(img_path, train_images_dir / img_path.name)

        label_name = img_path.stem + ".txt"
        label_path = source_labels_path / label_name
        if label_path.exists():
            shutil.copy2(label_path, train_labels_dir / label_name)
        else:
            (train_labels_dir / label_name).touch()

    print("正在复制验证集")
    for img_path in val_images:
        shutil.copy2(img_path, val_images_dir / img_path.name)

        label_name = img_path.stem + ".txt"
        label_path = source_labels_path / label_name
        if label_path.exists():
            shutil.copy2(label_path, val_labels_dir / label_name)
        else:
            (val_labels_dir / label_name).touch()

    classes_file = source_labels_path / "classes.txt"
    if classes_file.exists():
        shutil.copy2(classes_file, train_labels_dir / "classes.txt")
        shutil.copy2(classes_file, val_labels_dir / "classes.txt")
        print("\n已复制类别文件")

    print("\n数据集分割完成")
    print(f"输出目录: {output_path.absolute()}")
    print(f"训练集: {train_images_dir}")
    print(f"验证集: {val_images_dir}")


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    source_images = project_root / "dataset" / "insulator" / "images"
    source_labels = project_root / "dataset" / "insulator" / "labels"
    output_directory = project_root / "dataset" / "insulator_yolo"

    create_yolo_split(
        source_images_dir=str(source_images),
        source_labels_dir=str(source_labels),
        output_dir=str(output_directory),
        train_ratio=0.8,
        seed=42,
    )
