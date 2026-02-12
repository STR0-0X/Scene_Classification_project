import os
import numpy as np
import sys
from tqdm import tqdm
from sklearn.metrics import pairwise_distances_argmin
from utils import DATA_ROOT

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DESCRIPTORS_ROOT = os.path.join(BASE_DIR, "..", "outputs", "descriptors")

K = int(sys.argv[1])
VOCAB_PATH = os.path.join(BASE_DIR, "..", "outputs", "vocab", f"vocab_k{K}.npy")

HIST_ROOT = os.path.join(BASE_DIR, "..", "outputs", "histograms")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_vocab():
    vocab = np.load(VOCAB_PATH)
    return vocab


def encode_image(data, vocab):
    pts = data["pts"]
    desc = data["desc"]
    img_w, img_h = data["size"]

    k = vocab.shape[0]
    histograms = []

    # ---- Level 0: whole image ----
    if desc.shape[0] > 0:
        idx = pairwise_distances_argmin(desc, vocab)
        hist = np.bincount(idx, minlength=k).astype(np.float32)
    else:
        hist = np.zeros(k, dtype=np.float32)

    hist /= (hist.sum() + 1e-10)
    histograms.append(hist)

    # ---- Level 1: 2x2 grid ----
    for i in range(2):
        for j in range(2):
            x_min = j * img_w / 2
            x_max = (j + 1) * img_w / 2
            y_min = i * img_h / 2
            y_max = (i + 1) * img_h / 2

            mask = (
                (pts[:, 0] >= x_min) & (pts[:, 0] < x_max) &
                (pts[:, 1] >= y_min) & (pts[:, 1] < y_max)
            )

            if np.any(mask):
                idx = pairwise_distances_argmin(desc[mask], vocab)
                hist = np.bincount(idx, minlength=k).astype(np.float32)
            else:
                hist = np.zeros(k, dtype=np.float32)

            hist /= (hist.sum() + 1e-10)
            histograms.append(hist)

    # Concatenate all regions
    final_hist = np.concatenate(histograms)

    # Final L1 normalization
    final_hist /= (final_hist.sum() + 1e-10)

    return final_hist



def process_split(split_name, vocab):
    split_desc_root = os.path.join(DESCRIPTORS_ROOT, split_name)
    histograms = []
    labels = []
    class_names = sorted(os.listdir(split_desc_root))

    for label, cls in enumerate(class_names):
        cls_path = os.path.join(split_desc_root, cls)
        files = [f for f in os.listdir(cls_path) if f.endswith(".npy")]

        print(f"{split_name}/{cls}: {len(files)} images")

        for f in tqdm(files, leave=False):
            data = np.load(os.path.join(cls_path, f), allow_pickle=True).item()
            if data is None:
                continue
            hist = encode_image(data, vocab)


            histograms.append(hist)
            labels.append(label)

    return np.array(histograms), np.array(labels), class_names


def main():
    ensure_dir(HIST_ROOT)

    vocab = load_vocab()
    print(f"Loaded vocabulary: {vocab.shape}")

    X_train, y_train, class_names = process_split("train", vocab)
    X_test, y_test, _ = process_split("test", vocab)

    np.save(os.path.join(HIST_ROOT, f"train_hist_k{K}.npy"), X_train)
    np.save(os.path.join(HIST_ROOT, f"test_hist_k{K}.npy"), X_test)
    np.save(os.path.join(HIST_ROOT, f"train_labels_k{K}.npy"), y_train)
    np.save(os.path.join(HIST_ROOT, f"test_labels_k{K}.npy"), y_test)
    np.save(os.path.join(HIST_ROOT, "class_names.npy"), class_names)



    print("Histograms saved.")
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)


if __name__ == "__main__":
    main()
