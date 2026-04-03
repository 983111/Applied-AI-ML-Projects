# 🧠 Character-Level LSTM — Next Character Predictor

A **from-scratch** implementation of a character-level LSTM that predicts the
next characters given a text seed — built with pure NumPy (no PyTorch, no
TensorFlow). Every single gradient is computed by hand.

---

## 📐 Architecture

```
Input character
     │
     ▼
┌─────────────┐
│  Embedding  │  vocab_size → embed_dim          (learnable lookup table)
└──────┬──────┘
       │  x_t  (embed_dim,)
       ▼
┌─────────────────────────────────────────────────────┐
│                    LSTM Cell                        │
│                                                     │
│  z = W_x · x_t + W_h · h_{t-1} + b                │
│                                                     │
│  f = σ(z[0:H])          forget gate               │
│  i = σ(z[H:2H])         input  gate               │
│  g = tanh(z[2H:3H])     cell   gate (candidate)   │
│  o = σ(z[3H:4H])        output gate               │
│                                                     │
│  c_t = f ⊙ c_{t-1} + i ⊙ g                        │
│  h_t = o ⊙ tanh(c_t)                               │
└──────┬──────────────────────────────────────────────┘
       │  h_t  (hidden_size,)
       ▼
┌─────────────┐
│   Linear    │  hidden_size → vocab_size
└──────┬──────┘
       │  logits  (vocab_size,)
       ▼
   Softmax  →  probability over next character
```

**Training objective:** Cross-entropy loss over each time step, minimised with
Adam and Backpropagation Through Time (BPTT).

---

## 🚀 Quick Start

```bash
# 1. Train (500 epochs, ~1–2 min on modern CPU)
python train.py

# 2. More epochs / custom hyperparameters
python train.py --epochs 1000 --hidden_size 256 --lr 5e-4

# 3. Interactive inference
python inference.py                         # REPL
python inference.py --seed "hel" --temp 0.7

# 4. Visualise training loss
python visualize.py
```

---

## 🗂️ File Structure

```
char_lstm/
├── model.py        ← Core: LSTMCell, EmbeddingLayer, LinearDecoder, Adam
│                     (every gate + backprop implemented manually)
├── train.py        ← Training loop with BPTT, gradient clipping, Adam
├── inference.py    ← Load trained weights and generate text
├── visualize.py    ← ASCII + matplotlib loss curves
└── saved_model/
    ├── weights.npz        (NumPy arrays)
    ├── meta.json          (vocab, hyperparams)
    └── loss_history.json  (per-epoch loss)
```

---

## ⚙️ Hyperparameters

| Parameter     | Default | Notes                                      |
|---------------|---------|--------------------------------------------|
| `embed_dim`   | 16      | Character embedding dimensions             |
| `hidden_size` | 128     | LSTM hidden state size                     |
| `seq_len`     | 25      | BPTT truncation length                     |
| `lr`          | 1e-3    | Adam learning rate                         |
| `epochs`      | 500     | Training epochs                            |
| `temperature` | 0.7     | Sampling temperature (lower = more greedy) |

---

## 🔬 Key Concepts Demonstrated

### 1. Character Embeddings
Each character is mapped to a dense vector (`embed_dim`-dimensional) via a
learnable embedding matrix `E ∈ ℝ^{vocab × embed_dim}`. This is the same
mechanism used in word2vec and GPT, just at the character level.

### 2. LSTM Gates (Vanishing Gradient Solution)
Standard RNNs suffer from vanishing gradients over long sequences.  
LSTMs solve this with:
- **Forget gate** `f` — what to erase from cell state
- **Input gate** `i` — what new info to write
- **Cell gate** `g` — candidate values to write  
- **Output gate** `o` — what to expose as hidden state

The **cell state** `c` forms a gradient highway — gradients can flow
through many time steps with minimal decay.

### 3. Backpropagation Through Time (BPTT)
The sequence is unrolled and gradients flow backward through each time step.
Gradient clipping (`‖g‖ ≤ 5.0`) prevents exploding gradients.

### 4. Temperature Sampling
During inference, dividing logits by `temperature` before softmax controls
creativity vs. focus:
```
temp < 1.0  →  peaky distribution  →  predictable output
temp = 1.0  →  true model distribution
temp > 1.0  →  flat  distribution  →  creative / random output
```

---

## 📊 Expected Results

After ~500 epochs on this tiny toy corpus, the model mostly memorizes:
- Common word completions (`hel` → `hello how are you`)
- Basic grammar patterns
- Sentence structure from training corpus

Example outputs after training:
```
Input: 'hel'  →  'hello how are you doing today'
Input: 'how'  →  'how are you doing today'
Input: 'the'  →  'the quick brown fox jumps over'
Input: 'mac'  →  'machine learning is fascinating'
```

---

## 🧩 Extending the Project

- **Larger corpus:** Replace `CORPUS` in `train.py` with any `.txt` file
- **Word-level:** Change tokenisation from chars to words
- **Deeper model:** Stack two LSTMs (pass `h` of LSTM-1 as input to LSTM-2)
- **Attention:** Add a simple attention mechanism over hidden states
- **Beam search:** Replace greedy/temperature sampling with beam search
- **Web UI:** Wrap `inference.py` in a Flask endpoint + HTML input box

---

## 🛠️ Dependencies

```
numpy  ≥ 1.21   (only dependency for training + inference)
matplotlib      (optional, for loss curve plots)
```

No deep learning framework required.

---

## 💡 Why This Impresses

Building BPTT from scratch proves you understand:
1. How gradients flow through recurrent connections
2. Why LSTMs were invented (vanishing gradient problem)
3. The math behind every gate activation
4. How embeddings encode discrete symbols
5. Numerical stability tricks (log-sum-exp, gradient clipping)

This is the foundation of every modern language model — from LSTM to Transformer.
