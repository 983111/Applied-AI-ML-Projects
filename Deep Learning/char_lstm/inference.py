import argparse, json, numpy as np
from model import LSTMCell, EmbeddingLayer, LinearDecoder, softmax

def generate(seed, length=50, temp=0.7):
    with open("saved_model/meta.json") as f: meta = json.load(f)
    weights = np.load("saved_model/weights.npz")
    c2i, i2c = meta["vocab"], {v: k for k, v in meta["vocab"].items()}
    
    embed = EmbeddingLayer(meta["vocab_size"], 16)
    lstm = LSTMCell(16, 128)
    decoder = LinearDecoder(128, meta["vocab_size"])
    
    embed.E, lstm.W_x, lstm.W_h, lstm.b, decoder.W, decoder.b = weights['emb_E'], weights['lstm_W_x'], weights['lstm_W_h'], weights['lstm_b'], weights['dec_W'], weights['dec_b']

    h, c = np.zeros(128), np.zeros(128)
    # Prime the model with the seed
    for char in seed[:-1]:
        if char in c2i:
            h, c, _ = lstm.forward(embed.forward(c2i[char]), h, c)
    
    result = seed
    curr = seed[-1]
    for _ in range(length):
        if curr not in c2i: break
        h, c, _ = lstm.forward(embed.forward(c2i[curr]), h, c)
        p = softmax(decoder.forward(h) / temp)
        idx = np.random.choice(len(p), p=p)
        curr = i2c[idx]
        result += curr
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=str, default="hello")
    parser.add_argument("--temp", type=float, default=0.7)
    args = parser.parse_args()
    print(generate(args.seed, temp=args.temp))