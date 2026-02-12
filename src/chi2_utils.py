import numpy as np

def chi2_kernel(X, Y, eps=1e-10):
    """
    Compute the chi-squared kernel between two sets of histograms.
    X: (n_samples_X, n_features)
    Y: (n_samples_Y, n_features)
    Returns: (n_samples_X, n_samples_Y)
    """
    K = np.zeros((X.shape[0], Y.shape[0]), dtype=np.float32)

    for i, x in enumerate(X):
        num = (x - Y) ** 2
        denom = x + Y + eps
        K[i] = -0.5 * np.sum(num / denom, axis=1)

    return np.exp(K)
