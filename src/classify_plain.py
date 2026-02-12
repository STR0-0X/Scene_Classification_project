import os
import numpy as np
import matplotlib.pyplot as plt
import sys

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score

K = int(sys.argv[1])
C = float(sys.argv[2])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HIST_ROOT = os.path.join(BASE_DIR, "..", "outputs", "histograms")
RESULTS_ROOT = os.path.join(BASE_DIR, "..", "outputs", "results")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_confusion(cm, class_names, title, out_path):
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=90)
    plt.yticks(ticks, class_names)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(out_path)
    plt.close()


def main():
    ensure_dir(RESULTS_ROOT)

    X_train = np.load(os.path.join(HIST_ROOT, f"train_hist_plain_k{K}.npy"))
    y_train = np.load(os.path.join(HIST_ROOT, f"train_labels_plain_k{K}.npy"))
    X_test = np.load(os.path.join(HIST_ROOT, f"test_hist_plain_k{K}.npy"))
    y_test = np.load(os.path.join(HIST_ROOT, f"test_labels_plain_k{K}.npy"))

    DESCRIPTORS_ROOT = os.path.join(BASE_DIR, "..", "outputs", "descriptors", "train")
    class_names = sorted(os.listdir(DESCRIPTORS_ROOT))


    # ---- Nearest Neighbor ----
    print("Running Nearest Neighbor...")
    nn = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
    nn.fit(X_train, y_train)
    y_pred_nn = nn.predict(X_test)

    acc_nn = accuracy_score(y_test, y_pred_nn)
    cm_nn = confusion_matrix(y_test, y_pred_nn)

    print(f"NN accuracy: {acc_nn:.3f}")
    plot_confusion(
        cm_nn, class_names,
        "Nearest Neighbor Confusion Matrix",
        os.path.join(RESULTS_ROOT, f"confusion_nn_plain_k{K}.png")
    )

    # ---- Linear SVM ----
    print("Running Linear SVM...")
    svm = LinearSVC(C=C, max_iter=5000)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)

    acc_svm = accuracy_score(y_test, y_pred_svm)
    print(f"K={K}, C={C}, SVM_ACC={acc_svm}")


    cm_svm = confusion_matrix(y_test, y_pred_svm)

    print(f"Linear SVM accuracy: {acc_svm:.3f}")
    plot_confusion(
        cm_svm, class_names,
        "Linear SVM Confusion Matrix",
        os.path.join(RESULTS_ROOT, f"confusion_svm_plain_k{K}.png")

    )


if __name__ == "__main__":
    main()
