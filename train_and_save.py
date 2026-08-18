"""
train_and_save.py — Run this ONCE locally to train the CNN on real MNIST
and save the weights to model_weights.npz, which the HF Space app then
loads at startup (no training happens in the Space).

Usage (from inside cnn-foundation/, where mnist_data/ already exists):
    python train_and_save.py

Takes ~5-10 min on CPU for 5 epochs over full MNIST (60k samples).
Produces model_weights.npz alongside this script.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.engine import Tensor
from src.layers import Conv2D, MaxPool2D, Flatten, Linear
from src.activations import relu
from src.model import CNN
from src.losses import cross_entropy, accuracy
from src.optim import Adam
from src.utils import load_mnist, iterate_minibatches

np.random.seed(42)

# --- Load real MNIST (point at your local mnist_data directory) ----------
MNIST_DIR = os.path.join(os.path.dirname(__file__), "..", "examples", "mnist_data")
print(f"Loading MNIST from {MNIST_DIR} ...")
train_X, train_y, test_X, test_y = load_mnist(data_dir=MNIST_DIR)
print(f"  train: {train_X.shape}  test: {test_X.shape}")

# --- Build model ---------------------------------------------------------
model = CNN(
    Conv2D(in_channels=1, out_channels=8, kernel_size=3, pad=1), relu,
    MaxPool2D(pool_size=2),          # 28x28 -> 14x14
    Conv2D(in_channels=8, out_channels=16, kernel_size=3, pad=1), relu,
    MaxPool2D(pool_size=2),          # 14x14 -> 7x7
    Flatten(),
    Linear(in_features=16 * 7 * 7, out_features=10),
)

n_params = sum(p.data.size for p in model.parameters())
print(f"Total parameters: {n_params:,}")

# --- Training loop -------------------------------------------------------
EPOCHS      = 5
BATCH_SIZE  = 64
LR          = 0.001

optimizer = Adam(model.parameters(), lr=LR)
best_test_acc = 0.0

for epoch in range(EPOCHS):
    epoch_losses, epoch_accs = [], []

    for xb, yb in iterate_minibatches(train_X, train_y, batch_size=BATCH_SIZE, shuffle=True, seed=epoch):
        optimizer.zero_grad()
        logits = model(Tensor(xb))
        loss   = cross_entropy(logits, yb)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.data.item())
        epoch_accs.append(accuracy(logits, yb))

    # Eval on test set every epoch
    test_logits = model(Tensor(test_X))
    test_acc    = accuracy(test_logits, test_y)
    best_test_acc = max(best_test_acc, test_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"loss {np.mean(epoch_losses):.4f} | "
          f"train acc {np.mean(epoch_accs):.3f} | "
          f"test acc {test_acc:.3f}")

print(f"\nBest test accuracy: {best_test_acc:.3f}")

# --- Save weights --------------------------------------------------------
weights = {}
params = model.parameters()
for i, p in enumerate(params):
    weights[f"param_{i}"] = p.data

out_path = os.path.join(os.path.dirname(__file__), "model_weights.npz")
np.savez(out_path, **weights)
print(f"Weights saved → {out_path}  ({os.path.getsize(out_path) / 1024:.1f} KB)")