import os
import numpy as np
import sys
from tqdm import tqdm
from sklearn.cluster import KMeans

from utils import DATA_ROOT, IMAGE_EXTENSIONS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DESCRIPTORS_ROOT = os.path.join(BASE_DIR, "..", "outputs", "descriptors", "train")
VOCAB_ROOT = os.path.join(BASE_DIR, "..", "outputs", "vocab")

K = int(sys.argv[1]) if len(sys.argv) > 1 else 100
VOCAB_PATH = os.path.join(
    BASE_DIR, "..", "outputs", "vocab", f"vocab_k{K}.npy"
)

MAX_DESCRIPTORS = 50000 # total descriptors to sample


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_all_descriptor_files():
    desc_files = []
    for cls in os.listdir(DESCRIPTORS_ROOT):
        cls_path = os.path.join(DESCRIPTORS_ROOT, cls)
        if not os.path.isdir(cls_path):
            continue
        for f in os.listdir(cls_path):
            if f.endswith(".npy"):
                desc_files.append(os.path.join(cls_path, f))
    return desc_files


def sample_descriptors(desc_files, max_descriptors):
    sampled = []
    total = 0

    np.random.shuffle(desc_files)

    for path in tqdm(desc_files, desc="Sampling descriptors"):
        data = np.load(path, allow_pickle=True).item()
        desc = data["desc"]

        if desc.size == 0:
            continue

        sampled.append(desc)
        total += desc.shape[0]

        if total >= max_descriptors:
            break

    sampled = np.vstack(sampled)
    if sampled.shape[0] > max_descriptors:
        idx = np.random.choice(sampled.shape[0], max_descriptors, replace=False)
        sampled = sampled[idx]

    return sampled


def main():

    if os.path.exists(VOCAB_PATH):
        print(f"Vocab for k={K} already exists, skipping build.")
        return

    ensure_dir(VOCAB_ROOT)

    print("Collecting descriptor files...")
    desc_files = load_all_descriptor_files()
    print(f"Found {len(desc_files)} descriptor files")

    print(f"Sampling up to {MAX_DESCRIPTORS} descriptors...")
    descriptors = sample_descriptors(desc_files, MAX_DESCRIPTORS)
    print(f"Using {descriptors.shape[0]} descriptors")

    print(f"Running k-means with k={K} ...")
    kmeans = KMeans(
        n_clusters=K,
        random_state=0,
        verbose=1,
        n_init=10
    )
    kmeans.fit(descriptors)

    vocab = kmeans.cluster_centers_

    out_path = os.path.join(VOCAB_ROOT, f"vocab_k{K}.npy")
    np.save(out_path, vocab)

    print(f"Vocabulary saved to {out_path}")
    print(f"Vocabulary shape: {vocab.shape}")


if __name__ == "__main__":
    main()
