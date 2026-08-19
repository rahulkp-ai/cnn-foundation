# From ANN to CNN to RNN — Why Each Architecture Exists

> This note documents the conceptual progression across the three projects in this learning path:
> [ann-foundation](https://github.com/rahulkp-ai/ann-foundation) →
> [cnn-foundation](https://github.com/rahulkp-ai/cnn-foundation) →
> rnn-foundation (next)

---

## Part 1 — Limitations of ANN (Multi-Layer Perceptron)

An ANN (MLP) is a stack of fully-connected layers: every neuron in layer _n_
connects to every neuron in layer _n+1_. This is powerful for tabular data,
but breaks down badly on images and sequences.

### 1. No spatial awareness

A fully-connected layer flattens its input into a 1D vector before doing
anything. A 28×28 image becomes 784 numbers. The network has no idea that
pixel (3, 4) is next to pixel (3, 5) — spatial neighbourhoods are completely
destroyed on the way in.

**Consequence:** The network has to _relearn_ from scratch that nearby pixels
tend to be correlated. <b>It can do it, but it wastes enormous capacity doing so,
and it generalises poorly (a digit shifted one pixel to the right looks
completely different to a flat MLP).</b>

### 2. No translation invariance / equivariance

If a cat is in the top-left corner of one image and the bottom-right of
another, an MLP treats these as entirely different inputs. There is no
mechanism that says "the same pattern at different positions means the same
thing." Every position must be learned independently.

**Consequence:** Requires massive amounts of training data to cover all
possible positions, scales, and orientations of every pattern.

### 3. Parameter explosion on image inputs

A single fully-connected layer from a 224×224 RGB image (ImageNet size) to
1000 hidden units requires:

```
224 × 224 × 3 × 1000 = 150,528,000 parameters
```

Just the _first_ layer. This is computationally intractable, prone to
overfitting, and requires GPU memory that scales with image resolution squared.

### 4. No weight sharing

In an MLP, the weights used to detect an edge in the top-left corner are
completely separate from the weights used to detect an edge in the
bottom-right corner. The same feature must be detected by separate, independent
sets of weights at every position in the image.

**Consequence:** The model is wildly over-parameterised relative to the actual
structure of the problem.

### 5. Does not exploit the structure of the input domain

Images have three properties that ANNs completely ignore:

- **Local correlations** — nearby pixels are more related than distant ones
- **Hierarchical features** — edges → textures → parts → objects
- **Stationarity** — the same local feature (an edge, a curve) can appear anywhere

An MLP treats all these as irrelevant, learning a general function that happens
to work on images rather than a function _designed_ for image structure.

---

## Part 2 — How CNNs Overcome These Limitations

CNNs (Convolutional Neural Networks) introduce two core operations that
directly address every ANN limitation above:

### 1. Local connectivity → spatial awareness

Instead of connecting every input pixel to every neuron, a convolutional
filter only connects to a small local patch (e.g. 3×3). This means the
network _explicitly models the fact that nearby pixels are related_ — the
spatial structure of the input is preserved, not destroyed.

```
ANN:  neuron sees all 784 pixels at once
CNN:  each filter sees a 3×3 patch of 9 pixels at a time
```

### 2. Weight sharing → translation equivariance + massively fewer parameters

A single convolutional filter applies the _same weights_ at every position in
the image. One edge-detection filter with 9 weights detects edges everywhere,
rather than needing 784 separate edge-detectors (one per pixel position).

**Parameter count comparison (first layer, MNIST):**

| Architecture             | Parameters (first layer) |
| ------------------------ | ------------------------ |
| MLP (784 → 512)          | 784 × 512 = **401,408**  |
| CNN (1 → 8 filters, 3×3) | 8 × 9 + 8 bias = **80**  |

The CNN uses 5,000× fewer parameters to learn richer, more generalisable features.

### 3. Hierarchical feature learning

Stacking convolutional layers naturally builds a feature hierarchy:

```
Layer 1 filters:  edges, colour gradients         (local, primitive)
Layer 2 filters:  corners, curves, textures        (combinations of edges)
Layer 3 filters:  parts — eyes, wheels, loops     (combinations of textures)
Layer N filters:  objects — faces, digits, cats   (semantic concepts)
```

This matches how the visual cortex actually works (Hubel & Wiesel, 1962).
An MLP can theoretically learn this hierarchy too, but needs far more data
and parameters to do so because it gets no architectural inductive bias.

### 4. Pooling → translation invariance + spatial compression

Max pooling takes the maximum activation in a small spatial region, which:

- Makes the representation slightly invariant to small translations
  (a feature shifted by 1–2 pixels gives the same pooled output)
- Progressively compresses spatial dimensions (28→14→7→...), reducing
  computation and forcing the network to retain only the most salient features

### 5. Result: cnn-foundation on MNIST

The CNN in cnn-foundation has just **9,098 parameters** — far fewer than any
viable MLP — and achieves **98.6% test accuracy** on MNIST. A comparable MLP
typically needs 10–50× more parameters for similar performance, and generalises
worse to shifted/rotated digits.

---

## Part 3 — Limitations of CNN

Despite solving ANN's spatial problems, CNNs have their own fundamental
limitations that make them unsuitable for a whole class of problems.

### 1. Fixed-size input

A CNN expects inputs of a fixed spatial size (e.g. always 28×28). This is fine
for image classification but immediately breaks down for:

- **Variable-length sentences** ("the cat sat" vs. "the enormous black cat")
- **Time series** of different lengths
- **Audio** clips of different durations

You can pad inputs to a maximum length, but this is wasteful and introduces
artefacts.

### 2. No memory across time

A CNN processes its entire input simultaneously in one forward pass. It has no
concept of "what came before" in a sequence. Given the sentence:

```
"The bank by the river was steep."
"The bank refused my loan application."
```

The word "bank" means something completely different in each sentence. A CNN
classifying these would need to somehow capture long-range context within a
fixed-size window — and there's no principled way to do this.

### 3. No temporal / sequential dependencies

Images are spatially structured: nearby pixels are related. Language and
audio are _temporally_ structured: earlier tokens causally influence later ones.

```
"He said he would come, but he ___."   → "didn't" depends on everything before it
"Yesterday I was tired, so today ___." → "I'm rested" depends on the prior clause
```

CNNs can't model this causal, sequential dependency naturally. A 1D CNN can
capture local n-gram patterns in text, but long-range dependencies require
exponentially many stacked layers — impractical, and still not truly sequential.

### 4. Order invariance problem

A CNN with global average pooling (or a simple flatten) is approximately
_order-insensitive_: shuffling the input tokens changes the features somewhat,
but the lack of positional encoding means the architecture doesn't inherently
distinguish "dog bites man" from "man bites dog."

### 5. Cannot generate sequences autoregressively

A CNN produces a fixed-size output in one shot. It cannot generate a sequence
one token at a time, conditioning each new token on everything generated so far
— which is the fundamental operation needed for:

- Machine translation
- Text generation
- Speech synthesis
- Time series forecasting

### 6. No persistent state

A CNN processes each input independently. It has no memory of previous inputs.
Given a streaming sensor (stock prices, EEG signals, live audio), a CNN must
re-process a fixed window from scratch at every step — it cannot accumulate
evidence across time.

---

## Part 4 — Why We Need RNN

RNNs (Recurrent Neural Networks) are specifically designed to address every
CNN limitation above. The key insight is a single architectural addition:
**a hidden state that persists across time steps.**

```
ANN / CNN forward pass:
    output = f(input)           # stateless, input → output

RNN forward pass:
    h_t = f(input_t, h_{t-1})  # stateful: new hidden state depends on
    output_t = g(h_t)           #           current input AND previous state
```

### How RNNs solve each CNN limitation

| CNN Limitation                  | RNN Solution                                                                |
| ------------------------------- | --------------------------------------------------------------------------- |
| Fixed-size input                | Processes sequences of **any length**, one step at a time                   |
| No memory across time           | **Hidden state** `h_t` carries information from all previous steps          |
| No sequential dependencies      | Each output explicitly conditions on `h_{t-1}` (all prior context)          |
| Order insensitivity             | Processes tokens **left-to-right** (or right-to-left); order is fundamental |
| Can't generate autoregressively | Output at step _t_ feeds back as input at step _t+1_                        |
| No persistent state             | Hidden state persists across the entire sequence                            |

### What RNNs are naturally suited for

| Task                     | Why RNN fits                                            |
| ------------------------ | ------------------------------------------------------- |
| Language modelling       | Predict next word given all previous words              |
| Machine translation      | Encode a source sentence into a state; decode to target |
| Sentiment analysis       | Classify a sentence after reading it word by word       |
| Time series forecasting  | Predict next value given historical sequence            |
| Speech recognition       | Map audio frames (sequential) to characters             |
| Named entity recognition | Tag each word, conditioned on its left context          |

### The progression in one line each

```
ANN   — learns any function, but ignores input structure entirely
CNN   — exploits spatial structure via local connectivity + weight sharing
RNN   — exploits temporal/sequential structure via recurrent hidden state
```

Each architecture is not a replacement for the previous — they solve
fundamentally different structural properties of data:

- Tabular data → ANN
- Grid-structured data (images, audio spectrograms) → CNN
- Sequential / temporal data → RNN

---

## What's Coming: rnn-foundation

The next project in this series will build an RNN from scratch — the same
philosophy as ann-foundation and cnn-foundation:

- A `Tensor`-backed recurrent cell with hidden state
- Backpropagation Through Time (BPTT) implemented by hand
- Vanishing gradient problem — observed empirically, then fixed with LSTM/GRU
- Trained on a character-level language model (generate text from scratch)
- 52+ gradient-verified tests

The key new mathematical piece: **BPTT** unrolls the recurrence through time
and applies the chain rule across all time steps simultaneously — the temporal
analogue of spatial backprop through conv layers.

---

_Part of the [rahulkp-ai](https://github.com/rahulkp-ai) from-scratch ML series._
