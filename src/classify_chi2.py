import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from chi2_utils import chi2_kernel

print("ARGV:", sys.argv)

K = int(sys.argv[1])
C = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HIST_ROOT = os.path.join(BASE_DIR, "..", "outputs", "histograms")
RESULTS_ROOT = os.path.join(BASE_DIR, "..", "outputs", "results")
os.makedirs(RESULTS_ROOT, exist_ok=True)


# Load data
X_train = np.load(os.path.join(HIST_ROOT, f"train_hist_k{K}.npy"))
y_train = np.load(os.path.join(HIST_ROOT, f"train_labels_k{K}.npy"))
X_test  = np.load(os.path.join(HIST_ROOT, f"test_hist_k{K}.npy"))
y_test  = np.load(os.path.join(HIST_ROOT, f"test_labels_k{K}.npy"))

print("Computing chi2 kernel (train)...")
K_train = chi2_kernel(X_train, X_train)

print("Computing chi2 kernel (test)...")
K_test = chi2_kernel(X_test, X_train)

print("Training chi2-kernel SVM...")
svm = SVC(kernel="precomputed", C=C)
svm.fit(K_train, y_train)

y_pred = svm.predict(K_test)
acc = accuracy_score(y_test, y_pred)

print(f"K={K}, CHI2_SVM_ACC={acc}")
print(f"C={C}, ACC={acc}")

train_acc = accuracy_score(y_train, svm.predict(K_train))
print(f"Train ACC={train_acc}")

# Confusion matrix

cm = confusion_matrix(y_test, y_pred)
np.save(os.path.join(RESULTS_ROOT, f"confusion_chi2_k{K}.npy"), cm)

class_names = np.load(
    os.path.join(HIST_ROOT, "class_names.npy"),
    allow_pickle=True
)

plt.figure(figsize=(10,10))
plt.imshow(cm)
plt.colorbar()

ticks = np.arange(len(class_names))
plt.xticks(ticks, class_names, rotation=90)
plt.yticks(ticks, class_names)

plt.title("Chi-square SVM Confusion Matrix (Spatial Pyramid, k=400)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("outputs/results/confusion_chi2_k400.png")
plt.close()
