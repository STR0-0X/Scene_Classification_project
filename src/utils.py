import os

# Path to the data directory (relative to src/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "..", "data")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def count_images_in_class(class_path):
    return sum(
        1 for f in os.listdir(class_path)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    )


def inspect_split(split_name):
    split_path = os.path.join(DATA_ROOT, split_name)

    if not os.path.isdir(split_path):
        raise RuntimeError(f"Split not found: {split_path}")

    classes = sorted(
        d for d in os.listdir(split_path)
        if os.path.isdir(os.path.join(split_path, d))
    )

    print(f"\nSplit: {split_name}")
    print(f"Number of classes: {len(classes)}")

    total_images = 0
    for cls in classes:
        class_path = os.path.join(split_path, cls)
        n_images = count_images_in_class(class_path)
        total_images += n_images
        print(f"  {cls}: {n_images} images")

    print(f"Total images in {split_name}: {total_images}")


if __name__ == "__main__":
    inspect_split("train")
    inspect_split("test")
