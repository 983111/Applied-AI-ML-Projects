# Applied AI & ML Projects

A collection of hands-on projects exploring machine learning, deep learning, computer vision, and natural language processing. This repository brings together implementations built from scratch, model fine-tuning experiments, and complete training and evaluation workflows.

The projects are organised around understanding how models work, building them, and evaluating their behaviour—not just calling a pre-trained model.

## Projects

### 1. Neural Network from Scratch — NumPy

A four-layer fully connected neural network implemented using NumPy, without PyTorch, TensorFlow, or Keras for the model itself. It is trained on the MNIST handwritten-digit dataset.

**What it covers**
- Forward propagation through three ReLU hidden layers and a softmax output layer.
- Backpropagation implemented manually and checked with numerical gradient checking (relative error around 1e-8).
- Mini-batch stochastic gradient descent, learning-rate decay, and optional L2 regularisation.
- He weight initialisation, loss and accuracy tracking, confusion matrix, and per-class precision, recall, and F1.
- Saving and reloading model weights.

**Architecture**

`784 → 128 → 64 → 32 → 10`

**Files**
- [Open the NumPy notebook](Neural_network%20from%20scratch/neural_network_numpy.ipynb)
- [Project documentation](Neural_network%20from%20scratch/README.md)

### 2. CNN Image Classifier — CIFAR-10 (PyTorch)

A compact VGG-style convolutional neural network trained from scratch on CIFAR-10. The model has **1,197,578 parameters** and classifies images into the dataset's 10 categories.

**What it covers**
- Six convolutional layers arranged in three blocks, with BatchNorm, ReLU, and max pooling.
- Global average pooling and a dropout-based classification head.
- Random crop and horizontal-flip augmentation.
- SGD with Nesterov momentum, a one-cycle learning-rate schedule, weight decay, and label smoothing.
- Mixed-precision training on GPU.
- Evaluation using top-1 and top-3 accuracy, per-class metrics, a confusion matrix, and error analysis.
- Visualisation of learned filters and feature maps, plus an image prediction demo.

**Architecture**

`Input (3 × 32 × 32) → Conv blocks (64, 128, 256 channels) → Global Average Pooling → 192-unit head → 10 classes`

The notebook keeps a 5,000-image validation split separate from the official 10,000-image test set. The test set is reserved for final evaluation.

**Files**
- [Open the CNN notebook](ML-project%20CNN/cnn_cifar10_classifier.ipynb)
- [Project documentation](ML-project%20CNN/README_cnn_cifar10.md)
- [Confusion matrix](ML-project%20CNN/image/confusion_matrix.png)

### 3. Fine-Tuning DistilBERT — IT Support Classification

A text-classification project that fine-tunes `distilbert-base-uncased` to route IT-support questions into five support categories.

**Dataset and task**
- Uses real, human-written questions from Stack Exchange communities.
- Includes five categories: Linux, Apple, Android, Server & Network, and Security.
- Uses 20,000 questions in total (4,000 per category), with a stratified train/validation/test split.
- Stack Exchange community is used as the label. These questions are a proxy for support tickets, not a collection of manually labelled real IT tickets.

**What it covers**
- Text cleaning, de-duplication, and dataset preparation.
- TF-IDF with Logistic Regression as a baseline, including a keyword-masked variant.
- Fine-tuning DistilBERT with the Hugging Face Transformers library.
- Evaluation using accuracy, macro-F1, per-class metrics, a confusion matrix, and confident-error analysis.
- Inference on new support questions and model export.

**Files**
- [Open the DistilBERT notebook](Fine%20Tuning-%20DistilBert/distilbert_it_support_classification.ipynb)
- [Project documentation](Fine%20Tuning-%20DistilBert/README_distilbert.md)

### 4. Character-Level LSTM — Text Generation

A character-level recurrent neural network that learns patterns in text and generates sequences one character at a time.

**What it covers**
- LSTM-based sequence modelling.
- Character-level text preparation and generation.
- Training and inference scripts, with temperature-based sampling.
- Loss visualisation and saved model checkpoints.

**Files**
- [Project folder](Deep%20Learning/char_lstm/)
- [Project documentation](Deep%20Learning/char_lstm/README.md)

### 5. Prompt Engineering Lab

A small learning and experimentation space for understanding how prompt structure affects responses from large language models.

**What it covers**
- Zero-shot and few-shot prompting.
- Chain-of-thought and ReAct-style prompting concepts.
- Research notes and an interactive HTML prompt-testing interface.

**Files**
- [Prompt lab folder](prompt%20lab/)

---

## Technology Stack

| Area | Tools |
|---|---|
| Programming | Python |
| Numerical computing | NumPy |
| Deep learning | PyTorch |
| NLP and transformers | Hugging Face Transformers, Datasets, Tokenizers |
| Data handling | Pandas |
| Computer vision | Torchvision |
| Evaluation | Scikit-learn |
| Visualisation | Matplotlib |
| Experiment environment | Jupyter Notebook, Google Colab |

## Getting Started

Most notebook-based projects can be run in Google Colab. Open the relevant notebook from the project links above, select a GPU runtime where required, and run the cells in order.

For local execution, clone the repository:

```bash
git clone https://github.com/983111/Applied-AI-ML-Projects.git
cd Applied-AI-ML-Projects
```

Install the dependencies required by the individual project. Refer to its documentation for the specific setup and execution instructions.

## Repository Structure

```text
Applied-AI-ML-Projects/
├── Neural_network from scratch/
│   ├── neural_network_numpy.ipynb
│   └── README.md
├── ML-project CNN/
│   ├── cnn_cifar10_classifier.ipynb
│   ├── README_cnn_cifar10.md
│   ├── cifar_cnn.pt
│   └── image/
│       └── confusion_matrix.png
├── Fine Tuning- DistilBert/
│   ├── distilbert_it_support_classification.ipynb
│   ├── README_distilbert.md
│   └── model_files/
├── Deep Learning/
│   └── char_lstm/
│       └── README.md
└── prompt lab/
```

## Notes

- Reported architecture details and training configurations are documented in each project's README or notebook.
- Model performance values are intentionally not stated here unless supported by recorded experiment outputs. Run the notebooks to reproduce and record the results.
- Dataset licences and usage conditions should be checked before redistributing datasets or using them commercially.

## Author

**Vishwajeet Adkine**  
[Portfolio](https://prt-delta.vercel.app/) · [GitHub](https://github.com/983111)

## License

This repository is distributed under the MIT License. See the [LICENSE](LICENSE) file for details.
