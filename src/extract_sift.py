import os
import cv2
import numpy as np
from tqdm import tqdm
from utils import DATA_ROOT, IMAGE_EXTENSIONS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(BASE_DIR, "..", "outputs", "descriptors")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def extract_sift_from_image(image_path, sift):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    h, w = img.shape
    keypoints, descriptors = sift.detectAndCompute(img, None)

    if descriptors is None:
        return {
            "pts": np.empty((0, 2), dtype=np.float32),
            "desc": np.empty((0, 128), dtype=np.float32),
            "size": (w, h)
        }

    pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)

    return {
        "pts": pts,
        "desc": descriptors.astype(np.float32),
        "size": (w, h)
    }


def process_split(split_name):
    split_path = os.path.join(DATA_ROOT, split_name)
    out_split_path = os.path.join(OUTPUT_ROOT, split_name)

    ensure_dir(out_split_path)

    sift = cv2.SIFT_create()

    classes = sorted(
        d for d in os.listdir(split_path)
        if os.path.isdir(os.path.join(split_path, d))
    )

    for cls in classes:
        cls_path = os.path.join(split_path, cls)
        out_cls_path = os.path.join(out_split_path, cls)
        ensure_dir(out_cls_path)

        images = [
            f for f in os.listdir(cls_path)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        ]

        print(f"{split_name}/{cls}: {len(images)} images")

        for img_name in tqdm(images, leave=False):
            img_path = os.path.join(cls_path, img_name)
            data = extract_sift_from_image(img_path, sift)

            out_path = os.path.join(
                out_cls_path,
                os.path.splitext(img_name)[0] + ".npy"
            )

            np.save(out_path, data)



if __name__ == "__main__":
    process_split("train")
    process_split("test")
