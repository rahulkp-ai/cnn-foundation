> **CNN → logits → Cross-Entropy Loss → gradients → Adam → updated weights**

---

# 1. The Big Picture

Suppose your CNN receives an image:

```text
Image
  │
  ▼
Conv2D
  │
  ▼
ReLU
  │
  ▼
MaxPool
  │
  ▼
Conv2D
  │
  ▼
ReLU
  │
  ▼
MaxPool
  │
  ▼
Flatten
  │
  ▼
Linear
  │
  ▼
Logits
  │
  ▼
Cross-Entropy
  │
  ▼
Loss
  │
  ▼
Backpropagation
  │
  ▼
Gradients
  │
  ▼
Adam Optimizer
  │
  ▼
Updated weights
```

There are **three important concepts** here:

| Component         | Job                                        |
| ----------------- | ------------------------------------------ |
| **CNN**           | Extract features and make predictions      |
| **Cross-Entropy** | Measure how wrong the prediction is        |
| **Adam**          | Use gradients to improve the CNN's weights |

---

# 2. What Does the CNN Actually Output?

Imagine MNIST classification. You have:

```text
10 classes
0 1 2 3 4 5 6 7 8 9

```

Your final `Linear` layer might produce:

```text
logits = [2.1, 0.3, -1.2, 4.5, 0.7, -0.8, 1.2, 0.1, 0.5, -0.4]

```

These are **not probabilities**. They are called **logits**.

The largest value is:

```text
4.5 → class 3

```

So the network predicts **`3`**, but the true label might be **`7`**.

Now we need a mathematical way to answer:

> **"How bad was this prediction?"**

That's the job of **Cross-Entropy Loss**.

---

# 3. Softmax

Before understanding Cross-Entropy, you need Softmax. Softmax converts logits into probabilities.

For logits $z$:

$$p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

where:

- $z_i$ = logit for class $i$
- $p_i$ = probability of class $i$

The probabilities will sum to 1:

```text
Class     Probability
──────────────────────
  0          0.070
  1          0.010
  2          0.002
  3          0.750
  4          0.020
  5          0.004
  6          0.030
  7          0.010
  8          0.020
  9          0.006
──────────────────────
 Total       1.000

```

The CNN is basically saying:

> _"I think there's a 75% chance this image is class 3."_

---

# 4. Cross-Entropy Loss

Now suppose the correct class is **7**.

The probability assigned to the correct class is:

$$p_{\text{true}} = 0.01$$

Cross-Entropy is:

$$L = -\log(p_{\text{true}})$$

Therefore:

$$L = -\log(0.01) \approx 4.605$$

That's a large loss because the network was very confident about the wrong class.

---

# 5. Why $-\log()$?

This is one of the most important things to understand.

Consider:

| Probability assigned to correct class | Loss  |
| ------------------------------------- | ----- |
| **0.99**                              | 0.010 |
| **0.90**                              | 0.105 |
| **0.70**                              | 0.357 |
| **0.50**                              | 0.693 |
| **0.10**                              | 2.303 |
| **0.01**                              | 4.605 |
| **0.001**                             | 6.908 |

Notice something interesting:

### Correct and confident

```text
Prediction: Cat = 0.99
Loss ≈ 0.01 (Excellent)

```

### Wrong and confident

```text
Prediction: Cat = 0.001, Dog = 0.999
True: Cat
Loss ≈ 6.9 (Very bad)

```

Cross-Entropy strongly penalizes **confident wrong predictions**. That's exactly what we want.

---

# 6. Cross-Entropy in a CNN

For a single example:

$$L = -\log(p_y)$$

where $y$ is the correct class.

For a batch of $N$ images:

$$L = -\frac{1}{N}\sum_{i=1}^{N}\log(p_{i,y_i})$$

**Example:**

- Image 1 → correct probability = 0.9
- Image 2 → correct probability = 0.8
- Image 3 → correct probability = 0.6

$$L = -\frac{\log(0.9) + \log(0.8) + \log(0.6)}{3}$$

The result is the **batch loss**.

---

# 7. A Very Important Implementation Detail

In modern deep-learning frameworks, you usually **do not manually calculate Softmax first**.

Instead, you use:

```python
CrossEntropyLoss(logits, targets)

```

because Cross-Entropy can be calculated directly from logits in a numerically stable way.

Mathematically:

$$L = -z_y + \log\left(\sum_j e^{z_j}\right)$$

This is the **log-sum-exp formulation**.

So your CNN should conceptually do:

```text
CNN → Logits → Cross Entropy → Loss

```

rather than:

```text
CNN → Softmax → Cross Entropy → Loss

```

when implementing the training loss.

---

# 8. Now Comes Backpropagation

Suppose:

```text
CNN → Logits → Cross Entropy → Loss = 2.4

```

We now want to reduce `2.4` to something like `1.5` → `0.8` → `0.4` → `0.2`.

The optimizer needs to know:

> **"Which weights caused the loss to increase?"**

Backpropagation calculates:

$$\frac{\partial L}{\partial w}$$

for every parameter.

For example:

```text
weight      gradient
─────────────────────
  w1         +0.82
  w2         -0.14
  w3         +0.03
  w4         -1.21

```

These gradients tell us the direction in which each parameter affects the loss.

---

# 9. Vanilla Gradient Descent

The simplest optimizer is:

$$w_{t+1} = w_t - \eta g_t$$

where:

- $w_t$ = current weight
- $g_t$ = gradient
- $\eta$ = learning rate

Suppose:

- $w = 2.0$
- $\text{gradient} = 0.5$
- $\text{learning rate} = 0.1$

$$w_{\text{new}} = 2.0 - (0.1)(0.5) = 1.95$$

```text
Before: w = 2.00
After:  w = 1.95

```

Simple. But vanilla SGD has problems when handling different parameter scales.

---

# 10. Why Adam?

Imagine one CNN parameter has gradients:
`0.01, 0.02, 0.01, 0.03, 0.02`

Another parameter has:
`5.2, -3.8, 4.9, -5.1, 6.2`

Using the same raw gradient update for everything isn't ideal.

Adam adapts the update **for each parameter**.

Adam stands for **Adaptive Moment Estimation**. It combines ideas from:

1. **Momentum**
2. **Adaptive learning rates**

---

# 11. Adam's First Idea: Momentum

Adam keeps a moving average of gradients. This is called the **first moment**: $m_t$.

The basic equation is:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

Typical value: $\beta_1 = 0.9$

Suppose:

- $\text{previous momentum} = 0.5$
- $\text{current gradient} = 1.0$

$$m_t = 0.9(0.5) + 0.1(1.0) = 0.55$$

Instead of responding only to the current gradient, Adam remembers previous gradients. This gives the optimizer **momentum**.

---

# 12. Adam's Second Idea: Squared Gradients

Adam also tracks the moving average of squared gradients. This is the **second moment**: $v_t$.

Formula:

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

Typical value: $\beta_2 = 0.999$

Why square the gradient? Because `+5` and `-5` should both indicate a large gradient magnitude:

$$(+5)^2 = 25 \quad \text{and} \quad (-5)^2 = 25$$

So Adam knows how large the gradients tend to be.

---

# 13. Bias Correction

There's one more problem. Initially:

$$m_0 = 0, \quad v_0 = 0$$

So the moving averages are biased toward zero at the beginning. Adam corrects this using:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$

$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

These are called **bias-corrected moments**.

---

# 14. The Adam Update

Finally:

$$w_{t+1} = w_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

This is the core of Adam.

```text
                 momentum
                    │
                    ▼
                  m̂_t
w_{t+1} = w_t - η ──────────────
                 √v̂_t + ε
                    ▲
                    │
            gradient magnitude

```

---

# 15. Full Adam Algorithm

For every parameter $w$:

1. **Get gradient:**

$$g_t = \frac{\partial L}{\partial w}$$

2. **Update first moment:**

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

3. **Update second moment:**

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

4. **Bias correction:**

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

5. **Update parameter:**

$$w_{t+1} = w_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

---

# 16. How Adam and Cross-Entropy Work Together

This is the critical connection. Imagine your CNN produces:

```text
Logits → [1.2, 0.5, 3.8, -0.2]
Correct Class → 1

```

1. **Cross-Entropy says:** _"Your prediction is wrong. Loss = ..."_
2. **Backpropagation calculates:**

$$\frac{\partial \text{Loss}}{\partial \text{ConvWeights}}, \quad \frac{\partial \text{Loss}}{\partial \text{LinearWeights}}, \quad \frac{\partial \text{Loss}}{\partial \text{Biases}}$$

3. **Adam receives gradients:**

$$\text{gradients} \longrightarrow \text{Adam} \longrightarrow \text{updated weights}$$

4. The CNN performs another forward pass.

---

# 17. Complete Training Loop

Your CNN training essentially becomes:

```text
             ┌──────────────┐
             │    Image     │
             └──────┬───────┘
                    │
                    ▼
             ┌──────────────┐
             │     CNN      │
             └──────┬───────┘
                    │
                    ▼
                 Logits
                    │
                    ▼
          ┌──────────────────┐
          │  Cross-Entropy   │
          └────────┬─────────┘
                   │
                   ▼
                  Loss
                   │
                   ▼
            Backpropagation
                   │
                   ▼
               Gradients
                   │
                   ▼
            ┌──────────────┐
            │     Adam     │
            └──────┬───────┘
                   │
                   ▼
            Updated weights
                   │
                   └───────────┐
                               │
                               ▼
                         Next iteration

```

---

## 18. The Most Important Distinction

Don't confuse these two:

- **Cross-Entropy:** _"How wrong is my prediction?"_ $\rightarrow$ **Loss Function**
- **Adam:** _"Given the gradients, how should I change my parameters?"_ $\rightarrow$ **Optimizer**

They have completely different responsibilities.

---

## 19. Example With a Tiny CNN

Suppose your CNN is:

```text
Input → Conv2D → ReLU → MaxPool → Flatten → Linear → 10 logits

```

For one MNIST image:

- **True label:** `7`
- **CNN outputs:** `[1.2, 0.4, -0.8, 2.1, 0.5, -1.0, 0.2, 0.1, 0.7, -0.3]`
- **Largest logit:** `2.1 → class 3` _(Wrong)_

Softmax might produce:

- Class 3 → `0.55`
- Class 7 → `0.08`

$$L = -\log(0.08) \approx 2.53$$

Now backpropagation calculates gradients:

```text
Conv weight 1 →  0.12
Conv weight 2 → -0.04
Conv weight 3 →  0.83
...

```

Adam processes those gradients and updates the weights.

After thousands of updates, you might get:

- Class 3 → `0.02`
- Class 7 → `0.94`

$$L = -\log(0.94) \approx 0.062$$

The CNN has learned.

---

## 20. One Extremely Important Insight

The CNN does **not** directly learn: _"This is a 7."_

It learns parameters that minimize:

$$L = -\log(P(\text{correct class}))$$

Adam then helps minimize that objective.

So the learning process is fundamentally:

$$\text{Weights} \longrightarrow \text{Prediction} \longrightarrow \text{Loss} \longrightarrow \text{Gradient} \longrightarrow \text{Weight Update}$$

Repeated thousands of times.

---

## 21. Where This Fits in Your From-Scratch CNN

Given the CNN you've been building:

```text
Conv2D → ReLU → MaxPool2D → Conv2D → ReLU → MaxPool2D → Flatten → Linear → CrossEntropyLoss → Adam

```

Your architecture can be thought of as two halves:

### Forward Computation

```text
Conv2D → ReLU → MaxPool → Conv2D → ReLU → MaxPool → Flatten → Linear → Logits → Cross Entropy → Loss

```

### Learning

```text
Loss → Backpropagation → Gradients → Adam → Parameter updates

```

This distinction is **fundamental** when implementing a neural-network framework from scratch.

---

## Mental Model

Remember these four words:

> **CNN predicts → Cross-Entropy judges → Backprop calculates → Adam updates**

Or even shorter:

```text
CNN       = Think
Loss      = Judge
Gradient  = Explain
Adam      = Improve

```

Once you understand this, the next really important step is to derive **why the gradient of Softmax + Cross-Entropy simplifies to:**

$$\frac{\partial L}{\partial z_i} = p_i - y_i$$

because that equation connects your **final Linear layer → Cross-Entropy → backpropagation** beautifully. It's one of the most useful derivations to understand when building your CNN from scratch.
