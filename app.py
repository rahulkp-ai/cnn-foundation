"""
app.py — Gradio web interface for cnn-foundation.

Loads a pre-trained CNN (weights from model_weights.npz) and serves two
modes of interaction:
  1. Sketchpad: draw a digit (0–9) freehand with your mouse or touchscreen.
  2. Upload: upload any 28×28 or larger grayscale image of a digit.

The model was built entirely from scratch — no PyTorch, no TensorFlow,
just NumPy + a hand-written Tensor autograd engine. This demo is the
live proof it works.
"""

import os
import sys
import numpy as np
import gradio as gr
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(__file__))

from src.engine import Tensor
from src.layers import Conv2D, MaxPool2D, Flatten, Linear
from src.activations import relu
from src.model import CNN

# ---------------------------------------------------------------------------
# Model definition — must match train_and_save.py exactly.
# ---------------------------------------------------------------------------

def build_model():
    return CNN(
        Conv2D(in_channels=1, out_channels=8, kernel_size=3, pad=1), relu,
        MaxPool2D(pool_size=2),
        Conv2D(in_channels=8, out_channels=16, kernel_size=3, pad=1), relu,
        MaxPool2D(pool_size=2),
        Flatten(),
        Linear(in_features=16 * 7 * 7, out_features=10),
    )


def load_weights(model, path="model_weights.npz"):
    data = np.load(path)
    params = model.parameters()
    for i, p in enumerate(params):
        key = f"param_{i}"
        if key not in data:
            raise KeyError(f"Weight file missing key '{key}' — "
                           f"re-run train_and_save.py to regenerate weights.")
        p.data = data[key].astype(np.float64)
    return model


WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "model_weights.npz")
model = build_model()

if os.path.exists(WEIGHTS_PATH):
    model = load_weights(model, WEIGHTS_PATH)
    _weights_loaded = True
else:
    # Fallback: random weights — the Space will still run but won't classify correctly.
    # Prevents a hard crash if someone forks the repo without including weights.
    _weights_loaded = False
    print("[WARNING] model_weights.npz not found. Using random weights — run train_and_save.py locally to generate them.")

# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def preprocess(img_array: np.ndarray) -> np.ndarray:
    """
    Convert a raw image array (any size, any colour depth) into a
    normalised (1, 1, 28, 28) float64 array, MNIST-style:
      - Convert to grayscale
      - Invert if background is dark (sketchpad draws white on black;
        MNIST is black digit on white background — we auto-detect)
      - Resize to 28×28 with anti-aliasing
      - Normalize to [0, 1]
    """
    img = Image.fromarray(img_array.astype(np.uint8))

    # Handle RGBA (sketchpad outputs RGBA)
    if img.mode == "RGBA":
        # Alpha channel tells us where the user drew — composite onto white bg
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg.convert("RGB")

    img = img.convert("L")  # grayscale

    # MNIST digits are dark (low pixel value) on white background.
    # If the mean brightness is < 128, the background is dark → invert.
    arr = np.array(img, dtype=np.float32)
    if arr.mean() < 128:
        arr = 255.0 - arr

    img = Image.fromarray(arr.astype(np.uint8))
    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img, dtype=np.float64) / 255.0
    return arr.reshape(1, 1, 28, 28)


def predict(img_array: np.ndarray):
    """
    Run a preprocessed image through the CNN and return
    (label_string, confidence_dict_for_bar_chart).
    """
    if img_array is None:
        return "Draw or upload a digit first.", {}

    x = preprocess(img_array)
    logits = model(Tensor(x))

    # Softmax for display
    shifted = logits.data - logits.data.max()
    probs = np.exp(shifted) / np.exp(shifted).sum()
    probs = probs.flatten()

    pred_digit = int(np.argmax(probs))
    confidence = float(probs[pred_digit])

    label = f"**{pred_digit}**  ({confidence * 100:.1f}% confidence)"

    if not _weights_loaded:
        label += "\n\n⚠️  *Using random weights — run `train_and_save.py` to generate real weights.*"

    confidence_dict = {str(i): float(probs[i]) for i in range(10)}
    return label, confidence_dict


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

HEADER = """
# CNN Foundation — Digit Classifier

A **convolutional neural network built entirely from scratch** in pure Python + NumPy.
No PyTorch, no TensorFlow — just a hand-written `Tensor` autograd engine,
`im2col`-based convolution, max pooling, cross-entropy loss, and Adam optimizer.

Draw a digit (0–9) on the sketchpad below, or upload an image, then click **Classify**.

> **Source:** [github.com/rahulkp-ai/cnn-foundation](https://github.com/rahulkp-ai/cnn-foundation) &nbsp;·&nbsp;
> **Sister project:** [ann-foundation](https://github.com/rahulkp-ai/ann-foundation) (scalar autograd + MLP from scratch)
"""

EXAMPLES = [
    ["examples/sample_0.png"],
    ["examples/sample_1.png"],
    ["examples/sample_7.png"],
]

with gr.Blocks(title="CNN Foundation") as demo:
    gr.Markdown(HEADER)

    with gr.Tabs():
        # ---- Tab 1: Sketchpad ----
        with gr.TabItem("✏️  Draw"):
            with gr.Row():
                with gr.Column(scale=1):
                    sketchpad = gr.Sketchpad(
                        label="Draw a digit (0–9) here",
                        type="numpy",
                        brush=gr.Brush(default_size=18, colors=["#000000"]),
                        canvas_size=(280, 280),
                    )
                    with gr.Row():
                        draw_btn   = gr.Button("Classify", variant="primary")
                        clear_btn  = gr.ClearButton([sketchpad])

                with gr.Column(scale=1):
                    draw_label  = gr.Markdown("*Prediction appears here*")
                    draw_probs  = gr.Label(label="Class probabilities", num_top_classes=10)

            draw_btn.click(
                fn=predict,
                inputs=[sketchpad],
                outputs=[draw_label, draw_probs],
            )
            sketchpad.change(
                fn=predict,
                inputs=[sketchpad],
                outputs=[draw_label, draw_probs],
            )

        # ---- Tab 2: Upload ----
        with gr.TabItem("📁  Upload"):
            with gr.Row():
                with gr.Column(scale=1):
                    upload_img  = gr.Image(
                        label="Upload a digit image",
                        type="numpy",
                        height=280,
                    )
                    upload_btn  = gr.Button("Classify", variant="primary")

                with gr.Column(scale=1):
                    upload_label = gr.Markdown("*Prediction appears here*")
                    upload_probs = gr.Label(label="Class probabilities", num_top_classes=10)

            upload_btn.click(
                fn=predict,
                inputs=[upload_img],
                outputs=[upload_label, upload_probs],
            )

    # ---- How it works section ----
    with gr.Accordion("⚙️  How it works", open=False):
        gr.Markdown("""
### Architecture

```
Input (1×28×28)
→ Conv2D(1→8, 3×3, pad=1) → ReLU → MaxPool2D(2×2)   # 28×28 → 14×14
→ Conv2D(8→16, 3×3, pad=1) → ReLU → MaxPool2D(2×2)  # 14×14 → 7×7
→ Flatten → Linear(784→10)
→ CrossEntropy / Softmax
```

Total parameters: **9,098**

### What makes this different from a normal demo

Every layer in this CNN was implemented from mathematical first principles:

- **`Tensor` autograd engine** — reverse-mode automatic differentiation
  over NumPy arrays. Each operation records its parents and a `_backward`
  closure; `.backward()` walks a topological sort of the computation graph
  in reverse, accumulating gradients via the chain rule.
- **`im2col` convolution** — unrolls every receptive-field patch into a
  column so convolution becomes one matrix multiply. Fast *and* differentiable.
- **Max pooling backward** — gradient routes only to the max position in
  each window.
- **Adam optimizer** — implemented from the Kingma & Ba (2014) update
  equations, not borrowed from any library.
- **52 gradient tests** — every analytical gradient verified against
  central-difference numerical gradients.

Trained on MNIST (60,000 images) for 5 epochs with Adam (lr=0.001),
achieving **~97% test accuracy**.
        """)

    gr.Markdown(
        "<div style='text-align:center; color:#888; font-size:13px; margin-top:16px'>"
        "Built by <a href='https://github.com/rahulkp-ai' target='_blank'>Rahul KP</a> "
        "· MSc Computer Science, University of Calicut · 2026"
        "</div>"
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())