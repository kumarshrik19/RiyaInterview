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
    y_clean = logits.argmax(axis=1).astype(np.int64)

    y = y_clean.copy()
    flip = rng.random(n) < noise
    y[flip] = rng.integers(0, 3, size=flip.sum())
    return X, y, y_clean


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

        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        cache = (x, z1, h, logits, probs)
        return probs, cache

    def loss_and_grads(self, x, y):
        probs, cache = self.forward(x)
        x, z1, h, logits, probs = cache
        n = x.shape[0]

        loss = -np.log(probs[np.arange(n), y] + 1e-12).mean()

        dlogits = probs.copy()
        dlogits[np.arange(n), y] -= 1
        dlogits /= n

        dW2 = h.T @ dlogits
        db2 = dlogits.sum(axis=0)

        dh = dlogits @ self.W2.T
        dz1 = dh * (z1 > 0)

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
    return float((preds == y).mean())


def macro_f1_from_logits(logits, y, n_classes=3):
    preds = logits.argmax(axis=1)
    f1s = []
    for c in range(n_classes):
        tp = ((preds == c) & (y == c)).sum()
        fp = ((preds == c) & (y != c)).sum()
        fn = ((preds != c) & (y == c)).sum()

        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        f1s.append(f1)
    return float(np.mean(f1s))


def per_example_loss(logits, y):
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return -log_probs[np.arange(len(y)), y]


X, y, y_clean = make_dataset()

perm = np.random.permutation(len(X))
X = X[perm]
y = y[perm]
y_clean = y_clean[perm]

n_train = 700
X_train_raw, X_val_raw = X[:n_train], X[n_train:]
y_train, y_val = y[:n_train], y[n_train:]
y_val_clean = y_clean[n_train:]

mean = X_train_raw.mean(axis=0, keepdims=True)
std = X_train_raw.std(axis=0, keepdims=True).clip(min=1e-6)

X_train = (X_train_raw - mean) / std
X_val = (X_val_raw - mean) / std

print("train class counts:", np.bincount(y_train, minlength=3).tolist())
print("val class counts:", np.bincount(y_val, minlength=3).tolist())

model = NumpyMLP()
lr = 0.08
batch_size = 64

for epoch in range(200):
    order = np.random.permutation(len(X_train))

    for start in range(0, len(X_train), batch_size):
        idx = order[start:start + batch_size]
        xb, yb = X_train[idx], y_train[idx]

        loss, probs, grads = model.loss_and_grads(xb, yb)

        model.W1 -= lr * grads["W1"]
        model.b1 -= lr * grads["b1"]
        model.W2 -= lr * grads["W2"]
        model.b2 -= lr * grads["b2"]

    if epoch % 25 == 0 or epoch == 199:
        train_logits = model.logits(X_train)
        val_logits = model.logits(X_val)
        train_acc = accuracy_from_logits(train_logits, y_train)
        val_acc = accuracy_from_logits(val_logits, y_val)
        val_f1 = macro_f1_from_logits(val_logits, y_val)
        print(
            f"epoch={epoch:03d} "
            f"train_acc={train_acc:.4f} "
            f"val_acc={val_acc:.4f} "
            f"val_macro_f1={val_f1:.4f}"
        )


class TorchMLP(torch.nn.Module):
    def __init__(self, d_in=6, d_hidden=18, n_classes=3):
        super().__init__()
        self.fc1 = torch.nn.Linear(d_in, d_hidden)
        self.fc2 = torch.nn.Linear(d_hidden, n_classes)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


# Gradient check on one minibatch
batch_x = X_train[:32]
batch_y = y_train[:32]

torch_model = TorchMLP()
with torch.no_grad():
    torch_model.fc1.weight.copy_(torch.tensor(model.W1.T))
    torch_model.fc1.bias.copy_(torch.tensor(model.b1))
    torch_model.fc2.weight.copy_(torch.tensor(model.W2.T))
    torch_model.fc2.bias.copy_(torch.tensor(model.b2))

check_model = NumpyMLP()
check_model.W1 = model.W1.copy()
check_model.b1 = model.b1.copy()
check_model.W2 = model.W2.copy()
check_model.b2 = model.b2.copy()

loss_np, _, grads_np = check_model.loss_and_grads(batch_x, batch_y)

x_t = torch.tensor(batch_x, dtype=torch.float32)
y_t = torch.tensor(batch_y, dtype=torch.long)

torch_logits = torch_model(x_t)
torch_loss = torch.nn.functional.cross_entropy(torch_logits, y_t)
torch_loss.backward()

dW1_t = torch_model.fc1.weight.grad.detach().numpy().T
db1_t = torch_model.fc1.bias.grad.detach().numpy()
dW2_t = torch_model.fc2.weight.grad.detach().numpy().T
db2_t = torch_model.fc2.bias.grad.detach().numpy()

print("\nGradient check vs PyTorch autograd:")
print("max |dW1 diff| =", np.max(np.abs(grads_np["W1"] - dW1_t)))
print("max |db1 diff| =", np.max(np.abs(grads_np["b1"] - db1_t)))
print("max |dW2 diff| =", np.max(np.abs(grads_np["W2"] - dW2_t)))
print("max |db2 diff| =", np.max(np.abs(grads_np["b2"] - db2_t)))

# Top high-loss validation examples
val_logits = model.logits(X_val)
val_losses = per_example_loss(val_logits, y_val)
top5 = np.argsort(-val_losses)[:5]

print("\nTop-5 highest-loss validation examples:")
for rank, i in enumerate(top5, 1):
    pred = int(val_logits[i].argmax())
    print(
        rank,
        {
            "val_idx": int(i),
            "loss": round(float(val_losses[i]), 4),
            "label": int(y_val[i]),
            "pred": pred,
            "clean_label_for_verification_only": int(y_val_clean[i]),
        }
    )