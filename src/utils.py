"""
utils.py — MNIST loading, batching, accuracy, and visualization helpers.

Mirrors ann-foundation's `utils.py` role (dataset generation + decision
boundary plotting) one level up: real image data loading, minibatching
(MNIST is far too big to train on as one batch, unlike ann-foundation's
4-point toy dataset), and CNN-appropriate visualizations (training curves,
sample predictions, confusion matrix) in place of a 2D decision boundary.
"""

from __future__ import annotations
import gzip
import os
import struct
import urllib.request
import numpy as np

MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


def _read_idx_images(path_gz):
    with gzip.open(path_gz, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        buf = f.read(n * rows * cols)
        data = np.frombuffer(buf, dtype=np.uint8).reshape(n, 1, rows, cols)
    return data.astype(np.float64)


def _read_idx_labels(path_gz):
    with gzip.open(path_gz, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        buf = f.read(n)
        labels = np.frombuffer(buf, dtype=np.uint8)
    return labels.astype(np.int64)


def _download(url, dest):
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def _make_synthetic_mnist(n_train=2000, n_test=400, seed=0):
    """
    Fallback synthetic dataset used when real MNIST can't be downloaded
    (e.g. no internet access, or a blocked/offline environment).

    Generates 28x28 grayscale images of simple class-dependent blob/stroke
    patterns with noise — not a substitute for real MNIST accuracy numbers,
    but enough to exercise the full data pipeline, training loop, and every
    shape in the model end-to-end. Swap in `load_mnist()` whenever internet
    access to a real MNIST mirror is available.
    """
    rng = np.random.RandomState(seed)

    def make_split(n):
        images = np.zeros((n, 1, 28, 28), dtype=np.float64)
        labels = rng.randint(0, 10, size=n)
        for i, label in enumerate(labels):
            img = np.zeros((28, 28))
            cx, cy = 14 + (label - 5) * 0.8, 14
            radius = 4 + (label % 5)
            yy, xx = np.mgrid[0:28, 0:28]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 < radius ** 2
            img[mask] = 200
            img += rng.randn(28, 28) * 15
            images[i, 0] = np.clip(img, 0, 255)
        return images, labels

    train_images, train_labels = make_split(n_train)
    test_images, test_labels = make_split(n_test)
    return train_images, train_labels, test_images, test_labels


def load_mnist(data_dir="./mnist_data", normalize=True, use_synthetic_fallback=True):
    """
    Loads MNIST as (train_images, train_labels, test_images, test_labels).

    train_images / test_images : float64 arrays of shape (N, 1, 28, 28)
    train_labels / test_labels : int64 arrays of shape (N,)

    Attempts to download the standard MNIST .gz files into `data_dir` on
    first call (cached for subsequent calls). If downloading fails (e.g.
    no internet access) and `use_synthetic_fallback=True`, falls back to a
    synthetic dataset of the same shape so the rest of the pipeline
    (model, training loop, visualization) can still be exercised.
    """
    os.makedirs(data_dir, exist_ok=True)
    try:
        paths = {}
        for key, url in MNIST_URLS.items():
            fname = os.path.join(data_dir, os.path.basename(url))
            if not os.path.exists(fname):
                _download(url, fname)
            paths[key] = fname

        train_images = _read_idx_images(paths["train_images"])
        train_labels = _read_idx_labels(paths["train_labels"])
        test_images = _read_idx_images(paths["test_images"])
        test_labels = _read_idx_labels(paths["test_labels"])

    except Exception as e:
        if not use_synthetic_fallback:
            raise
        print(f"[load_mnist] Could not download real MNIST ({e}). "
              f"Falling back to a synthetic placeholder dataset of the same shape.")
        train_images, train_labels, test_images, test_labels = _make_synthetic_mnist()

    if normalize:
        train_images = train_images / 255.0
        test_images = test_images / 255.0

    return train_images, train_labels, test_images, test_labels


def iterate_minibatches(X, y, batch_size=32, shuffle=True, seed=None):
    """
    Yields (X_batch, y_batch) pairs covering the full dataset once.

    Equivalent in spirit to ann-foundation's simple `for x, y_true in data`
    loop, but for datasets too large to process as a single batch.
    """
    n = X.shape[0]
    indices = np.arange(n)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)

    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]


def plot_training_curve(losses, accuracies=None, save_path=None):
    """
    Plots loss (and optionally accuracy) over training steps/epochs.
    Analogous to ann-foundation's printed epoch/loss log, but visual.
    """
    import matplotlib.pyplot as plt

    if accuracies is not None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(losses, color="#d62728")
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Cross-Entropy Loss")
        axes[0].grid(alpha=0.3)

        axes[1].plot(accuracies, color="#1f77b4")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_ylim(0, 1.05)
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(losses, color="#d62728")
        ax.set_title("Training Loss")
        ax.set_xlabel("Step")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.grid(alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_sample_predictions(images, true_labels, pred_labels, n=12, save_path=None):
    """
    Plots a grid of sample digit images with predicted vs. true labels,
    coloring the title green if correct and red if wrong. Useful for a
    qualitative sanity check of what the CNN is getting right/wrong —
    the CNN analogue of ann-foundation's decision boundary plot.
    """
    import matplotlib.pyplot as plt

    n = min(n, images.shape[0])
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.2))
    axes = np.array(axes).reshape(-1)

    for i in range(rows * cols):
        ax = axes[i]
        ax.axis("off")
        if i < n:
            img = images[i, 0]
            correct = true_labels[i] == pred_labels[i]
            ax.imshow(img, cmap="gray")
            ax.set_title(f"pred={pred_labels[i]} true={true_labels[i]}",
                         color="#2ca02c" if correct else "#d62728", fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_confusion_matrix(true_labels, pred_labels, num_classes=10, save_path=None):
    """Plots a confusion matrix heatmap for classification results."""
    import matplotlib.pyplot as plt

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(true_labels, pred_labels):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_title("Confusion Matrix")
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
