# 1.2M-Parameter CNN for CIFAR-10

A compact VGG-style convolutional neural network (**1,197,578 parameters**) trained from scratch on CIFAR-10 with PyTorch.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/<your-repo>/blob/main/cnn_cifar10_classifier.ipynb)

> Replace `<your-username>` and `<your-repo>` after pushing to GitHub.

## Highlights

- Under 1.2M parameters, with the count printed and asserted in the notebook
- BatchNorm, He initialisation, global average pooling and a small dropout head
- Augmentation (random crop and flip), SGD with Nesterov momentum, one-cycle LR schedule, weight decay, label smoothing and mixed precision
- Clean train / validation / test protocol: the official test set is used only once, at the end
- Evaluation: curves, top-1 and top-3 accuracy, per-class report, confusion matrix, most-confused pairs, most confident mistakes
- Visualisation of first-layer filters and feature maps
- Upload your own image and get predictions (Colab)

## Architecture

```
Input 3x32x32
Block 1: Conv3x3(3->64)    BN ReLU  Conv3x3(64->64)    BN ReLU  MaxPool  -> 64  x16x16
Block 2: Conv3x3(64->128)  BN ReLU  Conv3x3(128->128)  BN ReLU  MaxPool  -> 128 x 8x 8
Block 3: Conv3x3(128->256) BN ReLU  Conv3x3(256->256)  BN ReLU  MaxPool  -> 256 x 4x 4
Global average pooling -> 256
Linear(256->192) ReLU Dropout(0.3) Linear(192->10)
```

| Part | Parameters |
|------|-----------:|
| 6 conv layers (no bias) | 1,144,512 |
| 6 BatchNorm layers | 1,792 |
| Classifier head | 51,274 |
| **Total** | **1,197,578** |

## Training Recipe

| Setting | Value |
|---------|-------|
| Data split | 45,000 train / 5,000 validation / 10,000 test |
| Augmentation | RandomCrop(32, padding 4), horizontal flip |
| Normalisation | per-channel CIFAR-10 mean and std |
| Optimiser | SGD, momentum 0.9, Nesterov |
| LR schedule | One-cycle, max LR 0.1, 20% warm-up |
| Weight decay | 5e-4 |
| Label smoothing | 0.1 |
| Epochs / batch size | 30 / 128 |
| Precision | mixed (fp16) on GPU |
| Model selection | best validation accuracy |

All values live in the configuration cell at the top of the notebook.

## How to Run

1. Open `cnn_cifar10_classifier.ipynb` in Google Colab and set **Runtime → Change runtime type → T4 GPU**.
2. Run **Runtime → Run all**. CIFAR-10 downloads automatically. Training time depends on the GPU Colab assigns you.
3. Fill in the results table below and add screenshots.

To run locally: `pip install torch torchvision matplotlib scikit-learn` (the "upload your own image" cells use `google.colab` and only work in Colab).

## Results

Fill in after running:

| Metric | Value |
|--------|-------|
| Test accuracy (top-1) | 92.23% |
| Test accuracy (top-3) | 98.77% |
| Best validation accuracy | 91.66% (reached in Epoch 30) |
| Training time (GPU used) | 8.5 minutes (using a Tesla T4 GPU) |

Add `![curves](images/curves.png)` and `![confusion matrix](images/confusion_matrix.png)` plus a short note on the most confused classes. Cats vs dogs is the usual weak spot on CIFAR-10, so check whether yours matches.

## Repository Layout

```
.
├── cnn_cifar10_classifier.ipynb
├── README.md
└── images/              # your curves and confusion matrix screenshots
```

## Ideas to Extend

- **Ablations:** remove augmentation, BatchNorm or label smoothing one at a time and report the accuracy drop
- CutMix or MixUp, residual connections, longer training
- Test-time augmentation (horizontal flip)
- Compare with ResNet-18 or a pretrained model
- Export to ONNX or TorchScript

## Requirements

Python 3.9+, PyTorch, torchvision, NumPy, Matplotlib, scikit-learn.

## Author

**Vishwajeet Adkine** — [Portfolio](https://prt-delta.vercel.app/)

## Licence

MIT. CIFAR-10 is by Krizhevsky, Nair and Hinton; see the [dataset page](https://www.cs.toronto.edu/~kriz/cifar.html) for terms and citation.
