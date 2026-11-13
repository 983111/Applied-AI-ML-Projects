# Four-Layer Neural Network from Scratch (NumPy)

A fully-connected neural network built with **only NumPy** — no deep-learning frameworks for the model. It implements forward propagation, backpropagation and mini-batch SGD, and is trained on handwritten-digit classification (MNIST).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/<your-repo>/blob/main/neural_network_numpy.ipynb)

> Replace `<your-username>` and `<your-repo>` in the badge link after you push this to GitHub.

## Highlights

- **Forward propagation** with ReLU hidden layers and a softmax output
- **Backpropagation** derived by hand and **verified with numerical gradient checking** (relative error ~1e-8)
- **Mini-batch SGD** with per-epoch shuffling, learning-rate decay and optional L2 regularisation
- **He initialisation** for stable ReLU training
- Evaluation: loss/accuracy curves, confusion matrix, per-class precision/recall/F1, misclassified examples
- Save and reload weights as a single `.npz` file

## Architecture

```
Input (784) → Dense 128 + ReLU → Dense 64 + ReLU → Dense 32 + ReLU → Dense 10 + Softmax
```

Four weight layers (three hidden, one output), about 109k parameters on MNIST.

## The Maths

For layer *l* with *a⁰ = X*:

```
z[l] = a[l-1] · W[l] + b[l]
a[l] = ReLU(z[l])            for hidden layers
a[L] = softmax(z[L])         for the output layer
```

Loss (cross-entropy plus optional L2 over a batch of *m* examples):

```
J = -(1/m) Σ y · log(a[L]) + (λ / 2m) Σ ||W[l]||²
```

Backpropagation (softmax + cross-entropy gives a clean output error):

```
δ[L] = (a[L] - y) / m
δ[l] = (δ[l+1] · W[l+1]ᵀ) ⊙ 1[z[l] > 0]
∂J/∂W[l] = a[l-1]ᵀ · δ[l] + (λ/m) · W[l]
∂J/∂b[l] = Σ δ[l]
```

SGD update: `W ← W − η · ∂J/∂W`, with η multiplied by a decay factor each epoch.

## Quick Start

### Option 1 — Google Colab (recommended)

1. Upload `neural_network_numpy.ipynb` to [Google Colab](https://colab.research.google.com/) (File → Upload notebook), or use the badge above once it's on GitHub.
2. Run **Runtime → Run all**.

No installation needed. MNIST is downloaded automatically.

### Option 2 — Run locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install numpy matplotlib tensorflow   # tensorflow is only used to download MNIST
jupyter notebook neural_network_numpy.ipynb
```

If TensorFlow isn't installed, the notebook falls back to scikit-learn's small 8×8 digits dataset (`pip install scikit-learn`).

## Hyper-parameters

Set in the `CONFIG` dictionary in the training cell:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `layer_sizes` | `[784, 128, 64, 32, 10]` | Units per layer |
| `epochs` | 30 | Passes over the training set |
| `batch_size` | 128 | Mini-batch size |
| `lr` | 0.1 | Initial learning rate |
| `lr_decay` | 0.95 | Multiplicative decay per epoch |
| `l2` | 1e-4 | L2 regularisation strength |

## Results

Fill this in after running the notebook on MNIST:

| Metric | Value |
|--------|-------|
| Test accuracy | _your result_ |
| Test loss | _your result_ |
| Training time (Colab CPU) | _your result_ |

Add a screenshot of your loss/accuracy curves and confusion matrix here, for example `![curves](images/curves.png)`.

## Notebook Structure

1. **Setup** — imports and seeding
2. **Data** — load, normalise, train/val/test split
3. **Network implementation** — `NeuralNetwork` class (`forward`, `backward`, `step`, `fit`)
4. **Gradient check** — numerical verification of backprop
5. **Training** — mini-batch SGD with logging
6. **Evaluation** — test metrics, confusion matrix, error analysis, batch-size experiment
7. **Save / load weights**

## Possible Extensions

- Momentum, RMSProp or Adam optimisers
- Dropout and batch normalisation
- Learning-rate schedules (step, cosine)
- Fashion-MNIST or CIFAR-10 (flattened)
- Vectorised hyper-parameter search

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- TensorFlow (optional, only to download MNIST) or scikit-learn (fallback dataset)

## Author

**Vishwajeet Adkine** — [Portfolio](https://prt-delta.vercel.app/)

## License

MIT — feel free to use and adapt.
