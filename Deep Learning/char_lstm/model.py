import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_grad(s):
    return s * (1 - s)

def tanh_grad(t):
    return 1 - t ** 2

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

class LSTMCell:
    def __init__(self, input_size: int, hidden_size: int):
        self.H, self.D = hidden_size, input_size
        # Xavier initialization for stability
        limit = np.sqrt(6 / (input_size + hidden_size))
        self.W_x = np.random.uniform(-limit, limit, (4 * self.H, self.D))
        self.W_h = np.random.uniform(-limit, limit, (4 * self.H, self.H))
        self.b = np.zeros(4 * self.H)

    def forward(self, x, h_prev, c_prev):
        z = self.W_x @ x + self.W_h @ h_prev + self.b
        f = sigmoid(z[:self.H])
        i = sigmoid(z[self.H:2*self.H])
        g = np.tanh(z[2*self.H:3*self.H])
        o = sigmoid(z[3*self.H:])
        c = f * c_prev + i * g
        tanh_c = np.tanh(c)
        h = o * tanh_c
        return h, c, (x, h_prev, c_prev, f, i, g, o, c, tanh_c)

    def backward(self, dh, dc, cache):
        x, h_prev, c_prev, f, i, g, o, c, tanh_c = cache
        do = dh * tanh_c
        dtanh_c = dh * o
        dc_total = dc + dtanh_c * tanh_grad(tanh_c)
        df, di, dg, dc_prev = dc_total * c_prev, dc_total * g, dc_total * i, dc_total * f
        dz = np.concatenate([df * sigmoid_grad(f), di * sigmoid_grad(i), 
                             dg * (1 - g**2), do * sigmoid_grad(o)])
        return self.W_x.T @ dz, self.W_h.T @ dz, dc_prev, {"W_x": np.outer(dz, x), "W_h": np.outer(dz, h_prev), "b": dz}

class EmbeddingLayer:
    def __init__(self, vocab_size, embed_dim):
        self.E = np.random.randn(vocab_size, embed_dim) * 0.01
    def forward(self, idx):
        self.last_idx = idx
        return self.E[idx].copy()
    def backward(self, d_embed):
        dE = np.zeros_like(self.E)
        dE[self.last_idx] = d_embed
        return dE

class LinearDecoder:
    def __init__(self, hidden_size, vocab_size):
        self.W = np.random.randn(vocab_size, hidden_size) * 0.01
        self.b = np.zeros(vocab_size)
    def forward(self, h):
        self.last_h = h
        return self.W @ h + self.b
    def backward(self, d_logits):
        return self.W.T @ d_logits, {"W": np.outer(d_logits, self.last_h), "b": d_logits}

class Adam:
    def __init__(self, lr=1e-3):
        self.lr, self.m, self.v, self.t = lr, {}, {}, 0
    def step(self, params, grads):
        self.t += 1
        for k in params:
            if k not in self.m: self.m[k], self.v[k] = np.zeros_like(params[k]), np.zeros_like(params[k])
            self.m[k] = 0.9 * self.m[k] + 0.1 * grads[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * (grads[k]**2)
            m_hat = self.m[k] / (1 - 0.9**self.t)
            v_hat = self.v[k] / (1 - 0.999**self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)