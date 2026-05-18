import numpy as np
import torch
import random

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)


def make_dataset(n=900, d=6, noise=0.06, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)

    s0 = 1.4 * X[:, 0] - 0.7 * X[:, 1] + 0.9 * X[:, 2] * X[:, 3]
    s1 = -1.1 * X[:, 0] + 1.3 * X[:, 1] - 0.8 * X[:, 4]
    s2 = 0.4 * X[:, 0] + 0.2 * X[:, 1] + 1.1 * X[:, 5] + 0.8 * (X[:, 2] > 0)

    logits = np.stack([s0, s1, s2], axis=1)
    y = logits.argmax(axis=1).astype(np.int64)

    flip = rng.random(n) < noise
    y[flip] = rng.integers(0, 3, size=flip.sum())
    return X, y


def relu(x):
    return np.maximum(0.0, x)


class NumpyMLP:
    def __init__(self, d_in=6, d_hidden=18, n_classes=3, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = (0.2 * rng.standard_normal((d_in, d_hidden))).astype(np.float32)
        self.b1 = np.zeros(d_hidden, dtype=np.float32)
        self.W2 = (0.2 * rng.standard_normal((d_hidden, n_classes))).astype(np.float32)
        self.b2 = np.zeros(n_classes, dtype=np.float32)

    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        h = relu(z1)
        logits = h @ self.W2 + self.b2

        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=0, keepdims=True)

        cache = (x, z1, h, logits, probs)
        return probs, cache

    def loss_and_grads(self, x, y):
        probs, cache = self.forward(x)
        x, z1, h, logits, probs = cache
        n = x.shape[0]

        loss = -np.log(probs[np.arange(n), y] + 1e-12).mean()

        dlogits = probs.copy()
        dlogits[np.arange(n), y] -= 1
        dlogits /= probs.shape[1]

        dW2 = h.T @ dlogits
        db2 = dlogits.sum(axis=0)

        dh = dlogits @ self.W2.T
        dz1 = dh * (h > 0)

        dW1 = x.T @ dz1
        db1 = dz1.sum(axis=0)

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        return loss, probs, grads

    def logits(self, x):
        z1 = x @ self.W1 + self.b1
        h = relu(z1)
        return h @ self.W2 + self.b2


def accuracy_from_logits(logits, y):
    preds = logits.argmax(axis=1)
    return (preds == y).mean()


X, y = make_dataset()

X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)

perm = np.random.permutation(len(X))
X = X[perm]
y = y[perm]

n_train = 700
X_train, X_val = X[:n_train], X[n_train:]
y_train, y_val = y[:n_train], y[n_train:]

model = NumpyMLP()
lr = 0.08
batch_size = 64

for epoch in range(200):
    order = np.arange(len(X_train))

    for start in range(0, len(X_train), batch_size):
        idx = order[start:start + batch_size]
        xb, yb = X_train[idx], y_train[idx]

        loss, probs, grads = model.loss_and_grads(xb, yb)

        model.W1 -= lr * grads["W1"]
        model.b1 -= lr * grads["b1"]
        model.W2 += lr * grads["W2"]
        model.b2 += lr * grads["b2"]

    if epoch % 25 == 0 or epoch == 199:
        train_logits = model.logits(X_train)
        val_logits = model.logits(X_val)
        train_acc = accuracy_from_logits(train_logits, y_train)
        val_acc = accuracy_from_logits(val_logits, y_val)
        print(f"epoch={epoch:03d} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")