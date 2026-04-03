import argparse
import json
import os

import numpy as np
from model import LSTMCell, EmbeddingLayer, LinearDecoder, Adam, softmax

CORPUS = """
hello how are you
hello world this is a test
how are you doing today
the quick brown fox jumps over the lazy dog
machine learning is fascinating
neural networks learn from data
deep learning uses multiple layers
recurrent networks handle sequences
long short term memory networks remember context
"""

def train(epochs=500, lr=1e-3, seq_len=25, embed_dim=16, hidden_size=128):
    text = CORPUS.strip().lower()
    chars = sorted(set(text))
    c2i = {c: i for i, c in enumerate(chars)}
    
    embed = EmbeddingLayer(len(chars), embed_dim)
    lstm = LSTMCell(embed_dim, hidden_size)
    decoder = LinearDecoder(hidden_size, len(chars))
    opt = Adam(lr=lr)
    loss_history = []
    
    indices = [c2i[c] for c in text]
    for epoch in range(1, epochs + 1):
        epoch_loss, n = 0, 0
        for start in range(0, len(indices) - seq_len - 1, seq_len // 2):
            chunk = indices[start: start + seq_len + 1]
            h, c = np.zeros(hidden_size), np.zeros(hidden_size)
            caches, logits_list, loss = [], [], 0
            # Forward
            for i in range(len(chunk)-1):
                e = embed.forward(chunk[i]); h, c, cache = lstm.forward(e, h, c)
                l = decoder.forward(h); p = softmax(l)
                loss -= np.log(p[chunk[i+1]] + 1e-12)
                caches.append(cache); logits_list.append((p, chunk[i+1]))
            # Backward
            dh, dc = np.zeros(hidden_size), np.zeros(hidden_size)
            grads = {k: np.zeros_like(v) for k, v in {"lstm_W_x": lstm.W_x, "lstm_W_h": lstm.W_h, "lstm_b": lstm.b, "dec_W": decoder.W, "dec_b": decoder.b, "emb_E": embed.E}.items()}
            for i in reversed(range(len(chunk)-1)):
                p, target = logits_list[i]; dl = p.copy(); dl[target] -= 1
                dh_dec, d_dec = decoder.backward(dl)
                grads["dec_W"] += d_dec["W"]; grads["dec_b"] += d_dec["b"]
                dx, dh, dc, d_lstm = lstm.backward(dh_dec + dh, dc, caches[i])
                grads["lstm_W_x"] += d_lstm["W_x"]; grads["lstm_W_h"] += d_lstm["W_h"]; grads["lstm_b"] += d_lstm["b"]
                grads["emb_E"][chunk[i]] += dx
            # Optimization
            for k in grads: grads[k] = np.clip(grads[k], -5, 5)
            opt.step({"lstm_W_x": lstm.W_x, "lstm_W_h": lstm.W_h, "lstm_b": lstm.b, "dec_W": decoder.W, "dec_b": decoder.b, "emb_E": embed.E}, grads)
            epoch_loss += loss / (len(chunk)-1); n += 1
        avg_loss = epoch_loss / max(1, n)
        loss_history.append(avg_loss)
        if epoch % 50 == 0:
            print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

    os.makedirs("saved_model", exist_ok=True)
    np.savez("saved_model/weights.npz", emb_E=embed.E, lstm_W_x=lstm.W_x, lstm_W_h=lstm.W_h, lstm_b=lstm.b, dec_W=decoder.W, dec_b=decoder.b)
    with open("saved_model/meta.json", "w") as f:
        json.dump(
            {
                "vocab": c2i,
                "vocab_size": len(chars),
                "embed_dim": embed_dim,
                "hidden_size": hidden_size,
                "seq_len": seq_len,
                "epochs": epochs,
                "lr": lr,
            },
            f,
        )
    with open("saved_model/loss_history.json", "w") as f:
        json.dump(loss_history, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq_len", type=int, default=25)
    parser.add_argument("--embed_dim", type=int, default=16)
    parser.add_argument("--hidden_size", type=int, default=128)
    args = parser.parse_args()
    train(
        epochs=args.epochs,
        lr=args.lr,
        seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
    )
