import os, json, numpy as np
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

def train(epochs=500, lr=1e-3, seq_len=25):
    text = CORPUS.strip().lower()
    chars = sorted(set(text))
    c2i = {c: i for i, c in enumerate(chars)}
    i2c = {i: c for i, c in enumerate(chars)}
    
    embed = EmbeddingLayer(len(chars), 16)
    lstm = LSTMCell(16, 128)
    decoder = LinearDecoder(128, len(chars))
    opt = Adam(lr=lr)
    
    indices = [c2i[c] for c in text]
    for epoch in range(1, epochs + 1):
        epoch_loss, n = 0, 0
        for start in range(0, len(indices) - seq_len - 1, seq_len // 2):
            chunk = indices[start: start + seq_len + 1]
            h, c = np.zeros(128), np.zeros(128)
            caches, logits_list, loss = [], [], 0
            # Forward
            for i in range(len(chunk)-1):
                e = embed.forward(chunk[i]); h, c, cache = lstm.forward(e, h, c)
                l = decoder.forward(h); p = softmax(l)
                loss -= np.log(p[chunk[i+1]] + 1e-12)
                caches.append(cache); logits_list.append((p, chunk[i+1]))
            # Backward
            dh, dc = np.zeros(128), np.zeros(128)
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
        if epoch % 50 == 0: print(f"Epoch {epoch} | Loss: {epoch_loss/n:.4f}")

    os.makedirs("saved_model", exist_ok=True)
    np.savez("saved_model/weights.npz", emb_E=embed.E, lstm_W_x=lstm.W_x, lstm_W_h=lstm.W_h, lstm_b=lstm.b, dec_W=decoder.W, dec_b=decoder.b)
    with open("saved_model/meta.json", "w") as f: json.dump({"vocab": c2i, "vocab_size": len(chars)}, f)

if __name__ == "__main__":
    train()