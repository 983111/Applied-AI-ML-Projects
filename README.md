# Stremini — implement

# Applied AI & ML Projects Ecosystem

Welcome to the **Applied AI & ML Projects** repository. This monorepo serves as a comprehensive portfolio and workspace for practical, end-to-end artificial intelligence and machine learning implementations. 

The projects span across classic Computer Vision (CNNs), deep Natural Language Processing (Transformers and LSTMs), fundamental algorithm design (Neural Networks from scratch), and modern LLM interaction techniques (Prompt Engineering).

## Table of Contents

- [Projects Overview](#projects-overview)
  - [1. Fine-Tuning DistilBERT (IT Support Classifier)](#1-fine-tuning-distilbert-it-support-classifier)
  - [2. Deep Learning: Character-Level LSTM](#2-deep-learning-character-level-lstm)
  - [3. ML Project: CNN Pipeline](#3-ml-project-cnn-pipeline)
  - [4. Neural Network from Scratch](#4-neural-network-from-scratch)
  - [5. Prompt Engineering Lab](#5-prompt-engineering-lab)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Detailed Features](#detailed-features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Running the Projects](#running-the-projects)
- [Contributing](#contributing)
- [License](#license)

## Projects Overview

### 1. Fine-Tuning DistilBERT (IT Support Classifier)
A complete NLP pipeline designed to automatically classify IT support tickets. By fine-tuning a pre-trained `DistilBERT` model from Hugging Face, this project demonstrates how to leverage transfer learning for domain-specific text classification tasks.

### 2. Deep Learning: Character-Level LSTM
A sequential deep learning model built to predict and generate text character by character. Utilizing Long Short-Term Memory (LSTM) networks, this project explores temporal data dependencies, vanishing gradients, and text generation techniques.

### 3. ML Project: CNN Pipeline
A highly modularized Convolutional Neural Network (CNN) pipeline for image classification. It encapsulates best practices in deep learning engineering, separating data loading, model architecture, training loops, and evaluation metrics into distinct, reusable modules.

### 4. Neural Network from Scratch
An educational deep-dive into the mathematics of Deep Learning. This project implements a fully functional multi-layer perceptron (MLP) without relying on heavy frameworks like PyTorch or TensorFlow, providing a transparent view of forward propagation, loss calculations, and backpropagation.

### 5. Prompt Engineering Lab
A research-oriented workspace focused on mastering interactions with Large Language Models (LLMs). It includes comprehensive research documentation and an interactive HTML lab to test and refine various prompting techniques.

## Project Structure

```text
.
├── Fine Tuning- DistilBert/
│   ├── data/                           # Raw and processed CSV datasets
│   ├── src/                            # Training, inference, and dataset creation scripts
│   ├── DistilBERT_IT_Support_Classifier.ipynb # Interactive notebook for fine-tuning
│   └── requirements.txt
│
├── Deep Learning/
│   └── char_lstm/
│       ├── model.py                    # LSTM architecture definition
│       ├── train.py                    # Training loop and backpropagation
│       ├── inference.py                # Text generation script
│       ├── visualize.py                # Loss curve visualizations
│       └── saved_model/                # Serialized weights (.npz) and metadata
│
├── ML-project CNN/
│   ├── src/
│   │   ├── data.py                     # Data augmentation and loaders
│   │   ├── model.py                    # CNN architecture
│   │   ├── train.py                    # Training pipeline
│   │   ├── evaluate.py                 # Evaluation metrics
│   │   └── visualize.py                # Feature map and result visualizer
│   └── requirements.txt
│
├── Neural_network from scratch/
│   └── nn_notebook.html                # Interactive notebook rendering the scratch implementation
│
└── prompt lab/
    ├── prompt-engineering-research.md  # Research notes on few-shot, chain-of-thought, etc.
    └── prompt-lab.html                 # Interactive prompt testing UI
```

## Technology Stack

- **Core Languages:** Python 3.9+, HTML/CSS
- **Deep Learning Frameworks:** PyTorch / TensorFlow / Keras (selected per module)
- **NLP Ecosystem:** Hugging Face Transformers, Datasets, Tokenizers
- **Data Processing & Math:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebooks

## Detailed Features

### Fine-Tuning DistilBERT
- **Custom Dataset Generation (`create_dataset.py`):** Tools to synthesize and clean IT support ticket data.
- **Transfer Learning (`train.py`):** Modifies the final classification head of DistilBERT for multi‑class classification.
- **Inference API (`inference.py`):** A lightweight script to pass raw text strings and receive support‑ticket category predictions.

### Character-Level LSTM
- **Custom Architecture (`model.py`):** A pure LSTM implementation mapping one‑hot encoded characters to hidden states.
- **Stateful Text Generation (`inference.py`):** Utilizes temperature scaling to control the randomness/creativity of the generated text.
- **Model Checkpointing:** Saves model weights (`weights.npz`) and loss histories (`loss_history.json`) for seamless experiment tracking.

### CNN Modular Pipeline
- **Separation of Concerns:** Deep‑learning best practices are applied by decoupling model, data, training, and evaluation logic.
- **Visualization Tools (`visualize.py`):** Scripts to plot training/validation loss curves and visualize learned feature maps.

### Prompt Lab
- **Theoretical Frameworks:** Detailed markdown files breaking down Zero‑shot, Few‑shot, Chain‑of‑Thought (CoT), and ReAct prompting.
- **Interactive HTML:** A zero‑setup HTML file to simulate and structure prompts before sending them to production LLM APIs.

## Getting Started

### Prerequisites
Ensure you have Python 3.9+ installed. It is highly recommended to use a virtual environment (venv or conda).

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Running the Projects

#### 1. Fine‑Tuning DistilBERT
```bash
cd "Fine Tuning- DistilBert"
pip install -r requirements.txt

# Train the model via script
python src/train.py

# Or open the Jupyter Notebook to run step‑by‑step
jupyter notebook DistilBERT_IT_Support_Classifier.ipynb
```

#### 2. Deep Learning (Char LSTM)
```bash
cd "Deep Learning/char_lstm"

# Train the LSTM
python train.py

# Generate new text
python inference.py

# View loss curves
python visualize.py
```

#### 3. CNN ML Project
```bash
cd "ML-project CNN"
pip install -r requirements.txt

# Run the end‑to‑end pipeline
python src/train.py
python src/evaluate.py
```

#### 4. Educational Notebooks & HTML Labs
No heavy installation is required for the HTML files. Simply open them in any modern web browser:

- **macOS:** `open "prompt lab/prompt-lab.html"`
- **Linux:** `xdg-open "Neural_network from scratch/nn_notebook.html"`
- **Windows:** Double‑click the file in Windows Explorer.

## Contributing
Contributions, issues, and feature requests are welcome!

1. **Fork** the project.
2. **Create** your feature branch: `git checkout -b feature/AmazingModel`.
3. **Commit** your changes: `git commit -m 'Add some AmazingModel'`.
4. **Push** to the branch: `git push origin feature/AmazingModel`.
5. **Open** a Pull Request.

## License
Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
