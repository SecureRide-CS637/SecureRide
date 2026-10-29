#!/usr/bin/env python3
"""
SecureRide predictors (Kim et al., ACM TECS 24(5s), Sec. 4.2 and 6.1)

Unified API supporting three backends:
  backend="torch"    ResNet-based 1D CNN with channel attention (paper Sec. 6.1)
                     or GRU baseline. Fast, native PyTorch CPU/GPU.
  backend="keras"    ResNet-based 1D CNN or GRU using TensorFlow >= 2.13.
  backend="sklearn"  HistGradientBoosting on hand-crafted window features.
                     Fast CPU baseline requiring no deep learning framework.

Predictor API (used by train.py, dualmode.py, and predict_sample.py):
    p = make_predictor(backend, L, hl, arch="cnn")
    history = p.fit(Xtr, ytr, Xva, yva)       X: (N, L, 3) normalised [V, I, C];  y: (N,) 0/1
    preds = p.predict(X)                     -> (N,) int8 (0 = legal, 1 = illegal)
    probs = p.predict_proba(X)               -> (N, 2) float32 class probabilities
    p.save(stem)                             -> saves <stem>.<pt|joblib|keras>
    p = load_predictor(stem_or_path)
    ms = p.exec_ms()                         -> mean single-window inference latency
    n = p.n_params()                         -> trainable parameter count
"""
import os, time
import numpy as np

def resolve_class_weight(cw, y):
    """class_weight option -> np.array([w_legal, w_illegal]) or None.
    cw: None | "balanced" | (w0, w1).  "balanced" = N / (2 * n_c)  (sklearn convention)."""
    if cw is None or cw == "none":
        return None
    y = np.asarray(y)
    if isinstance(cw, str):
        assert cw == "balanced", "class_weight must be None, 'balanced' or (w0, w1)"
        n = len(y); n1 = max(int((y == 1).sum()), 1); n0 = max(n - int((y == 1).sum()), 1)
        return np.array([n / (2.0 * n0), n / (2.0 * n1)], np.float64)
    return np.asarray(cw, np.float64)


# ===================================================================== PyTorch Backend (Paper Architecture)
def _torch():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    return torch, nn, optim, TensorDataset, DataLoader


class PyTorchResNet1D(object):
    """Factory creating nn.Module for 1D ResNet with Channel Attention."""
    @staticmethod
    def create(L, hl, in_channels=3, filters=(16, 32, 64), kernel=3, dropout=0.3):
        torch, nn, _, _, _ = _torch()

        class AttentionBlock(nn.Module):
            def __init__(self, channels=3, hidden=8):
                super().__init__()
                self.fc1 = nn.Linear(channels, hidden)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(hidden, channels)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                # x shape: (B, C, L)
                s = x.mean(dim=2)               # Global Average Pooling -> (B, C)
                s = self.fc1(s)
                s = self.relu(s)
                s = self.fc2(s)
                s = self.sigmoid(s).unsqueeze(2) # (B, C, 1)
                return x * s

        class ResidualBlock(nn.Module):
            def __init__(self, in_f, out_f, k=3, drop=0.3):
                super().__init__()
                pad = k // 2
                self.conv1 = nn.Conv1d(in_f, out_f, kernel_size=k, padding=pad)
                self.bn1 = nn.BatchNorm1d(out_f)
                self.relu1 = nn.ReLU()
                self.drop1 = nn.Dropout(drop)

                self.conv2 = nn.Conv1d(out_f, out_f, kernel_size=k, padding=pad)
                self.bn2 = nn.BatchNorm1d(out_f)
                self.relu2 = nn.ReLU()
                self.drop2 = nn.Dropout(drop)

                self.conv3 = nn.Conv1d(out_f, out_f, kernel_size=k, padding=pad)
                self.bn3 = nn.BatchNorm1d(out_f)

                if in_f != out_f:
                    self.shortcut = nn.Sequential(
                        nn.Conv1d(in_f, out_f, kernel_size=1),
                        nn.BatchNorm1d(out_f)
                    )
                else:
                    self.shortcut = nn.BatchNorm1d(out_f)
                self.relu_out = nn.ReLU()

            def forward(self, x):
                shortcut = self.shortcut(x)
                y = self.conv1(x)
                y = self.bn1(y)
                y = self.relu1(y)
                y = self.drop1(y)

                y = self.conv2(y)
                y = self.bn2(y)
                y = self.relu2(y)
                y = self.drop2(y)

                y = self.conv3(y)
                y = self.bn3(y)

                return self.relu_out(y + shortcut)

        class ResNet1DModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.att = AttentionBlock(channels=in_channels, hidden=8)
                self.blocks = nn.ModuleList()
                curr_f = in_channels
                for b in range(hl):
                    out_f = filters[b]
                    self.blocks.append(ResidualBlock(curr_f, out_f, k=kernel, drop=dropout))
                    curr_f = out_f
                self.gap = nn.AdaptiveAvgPool1d(1)
                self.head = nn.Linear(curr_f, 2)

            def forward(self, x):
                # input x: (B, L, C) -> transpose to (B, C, L)
                x = x.transpose(1, 2)
                x = self.att(x)
                for blk in self.blocks:
                    x = blk(x)
                x = self.gap(x).squeeze(2)      # (B, curr_f)
                out = self.head(x)              # (B, 2) logits
                return out

        return ResNet1DModel()


class PyTorchGRU(object):
    """Factory creating nn.Module for Stacked GRU baseline (paper Sec. 6.1)."""
    @staticmethod
    def create(L, hl, in_channels=3, hidden_size=32, dropout=0.3):
        torch, nn, _, _, _ = _torch()

        class GRUModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = nn.GRU(
                    input_size=in_channels,
                    hidden_size=hidden_size,
                    num_layers=hl,
                    batch_first=True,
                    dropout=dropout if hl > 1 else 0.0
                )
                self.head = nn.Linear(hidden_size, 2)

            def forward(self, x):
                # x shape: (B, L, C)
                out, _ = self.gru(x)
                # Take last timestep representation
                last_step = out[:, -1, :]
                logits = self.head(last_step)
                return logits

        return GRUModel()


class TorchPredictor:
    backend = "torch"

    def __init__(self, L, hl, arch="cnn", epochs=60, batch=256, lr=1e-3, patience=10, l2=1e-3, seed=0, verbose=2,
                 class_weight=None):
        self.class_weight = class_weight          # None | "balanced" | (w_legal, w_illegal)
        self.L, self.hl, self.arch = L, hl, arch
        self.epochs, self.batch, self.lr, self.patience, self.l2, self.seed, self.verbose = \
            epochs, batch, lr, patience, l2, seed, verbose
        self.model = None
        self.device = None

    def _init_model(self):
        torch, nn, _, _, _ = _torch()
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if self.arch == "cnn":
            self.model = PyTorchResNet1D.create(self.L, self.hl, in_channels=3, dropout=0.3)
        else:
            self.model = PyTorchGRU.create(self.L, self.hl, in_channels=3, hidden_size=32, dropout=0.3)
        self.model.to(self.device)

    def fit(self, Xtr, ytr, Xva, yva):
        torch, nn, optim, TensorDataset, DataLoader = _torch()
        self._init_model()

        ds_tr = TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(ytr).long())
        ds_va = TensorDataset(torch.from_numpy(Xva).float(), torch.from_numpy(yva).long())

        train_loader = DataLoader(ds_tr, batch_size=self.batch, shuffle=True)
        val_loader = DataLoader(ds_va, batch_size=min(4096, len(yva)), shuffle=False)

        w = resolve_class_weight(self.class_weight, ytr)
        criterion = nn.CrossEntropyLoss(weight=None if w is None else torch.tensor(w, dtype=torch.float32, device=self.device))
        # validation loss (used for early stopping / LR schedule) is UNweighted on purpose: it is
        # computed on the validation set, which has the same class prior as the data you will deploy on.
        crit_val = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.l2)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-5)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss, correct, total = 0.0, 0, 0
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                logits = self.model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(by)
                preds = logits.argmax(dim=1)
                correct += (preds == by).sum().item()
                total += len(by)

            train_loss = total_loss / total
            train_acc = correct / total

            # Validation
            self.model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    logits = self.model(bx)
                    loss = crit_val(logits, by)
                    val_loss += loss.item() * len(by)
                    preds = logits.argmax(dim=1)
                    val_correct += (preds == by).sum().item()
                    val_total += len(by)

            val_loss = val_loss / val_total
            val_acc = val_correct / val_total
            scheduler.step(val_loss)

            history["loss"].append(train_loss)
            history["accuracy"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)

            if self.verbose >= 2 and (epoch % 5 == 0 or epoch == 1 or epoch == self.epochs):
                print(f"Epoch {epoch:02d}/{self.epochs:02d} - loss: {train_loss:.4f} - acc: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")

            # Early stopping check
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                import copy
                best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if self.verbose >= 1:
                        print(f"Early stopping at epoch {epoch} (best val_loss: {best_val_loss:.4f})")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return history

    def predict(self, X):
        torch, _, _, _, _ = _torch()
        self.model.eval()
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=np.float32)
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

        out_preds = []
        batch_size = 4096
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                bx = torch.from_numpy(X[i:i + batch_size]).float().to(self.device)
                logits = self.model(bx)
                preds = logits.argmax(dim=1).cpu().numpy().astype(np.int8)
                out_preds.append(preds)
        return np.concatenate(out_preds) if out_preds else np.zeros((0,), dtype=np.int8)

    def predict_proba(self, X):
        torch, _, _, _, _ = _torch()
        self.model.eval()
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=np.float32)
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

        out_probs = []
        batch_size = 4096
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                bx = torch.from_numpy(X[i:i + batch_size]).float().to(self.device)
                logits = self.model(bx)
                probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
                out_probs.append(probs)
        return np.concatenate(out_probs, axis=0) if out_probs else np.zeros((0, 2), dtype=np.float32)

    def n_params(self):
        return int(sum(p.numel() for p in self.model.parameters() if p.requires_grad))

    def exec_ms(self, n=100):
        torch, _, _, _, _ = _torch()
        self.model.eval()
        x = torch.zeros((1, self.L, 3), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            # Warm-up
            for _ in range(5):
                _ = self.model(x)
            t = time.perf_counter()
            for _ in range(n):
                _ = self.model(x)
            return (time.perf_counter() - t) / n * 1e3

    def save(self, stem):
        torch, _, _, _, _ = _torch()
        path = stem if stem.endswith(".pt") else stem + ".pt"
        payload = {
            "model_state": self.model.state_dict(),
            "L": self.L,
            "hl": self.hl,
            "arch": self.arch,
            "backend": "torch",
            "seed": self.seed
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path):
        torch, _, _, _, _ = _torch()
        payload = torch.load(path, map_location="cpu")
        o = cls(payload["L"], payload["hl"], arch=payload.get("arch", "cnn"), seed=payload.get("seed", 0))
        o._init_model()
        o.model.load_state_dict(payload["model_state"])
        o.model.eval()
        return o


# ===================================================================== Keras (paper architecture)
def _tf():
    import tensorflow as tf
    return tf


def build_resnet1d(L, hl, filters=(16, 32, 64), kernel=3, dropout=0.3, l2=1e-3):
    """ResNet-based 1D CNN, paper Sec. 6.1."""
    assert 1 <= hl <= 3
    tf = _tf(); layers = tf.keras.layers; reg = tf.keras.regularizers.l2(l2)

    inp = layers.Input(shape=(L, 3), name="vic")
    s = layers.GlobalAveragePooling1D()(inp)
    s = layers.Dense(8, activation="relu")(s)
    s = layers.Dense(3, activation="sigmoid")(s)
    x = layers.Multiply()([inp, layers.Reshape((1, 3))(s)])

    for b in range(hl):
        f = filters[b]
        sc = x
        y = x
        for _ in range(2):
            y = layers.Conv1D(f, kernel, padding="same", kernel_regularizer=reg)(y)
            y = layers.BatchNormalization()(y)
            y = layers.ReLU()(y)
            y = layers.Dropout(dropout)(y)
        y = layers.Conv1D(f, kernel, padding="same", kernel_regularizer=reg)(y)
        y = layers.BatchNormalization()(y)
        if sc.shape[-1] != f:
            sc = layers.Conv1D(f, 1, padding="same", kernel_regularizer=reg)(sc)
        sc = layers.BatchNormalization()(sc)
        x = layers.ReLU()(layers.Add()([y, sc]))

    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(2, activation="softmax")(x)
    return tf.keras.Model(inp, out, name=f"resnet1d_L{L}_HL{hl}")


def build_gru(L, hl, units=32, dropout=0.3, l2=1e-3):
    """GRU baseline, paper Sec. 6.1."""
    tf = _tf(); layers = tf.keras.layers; reg = tf.keras.regularizers.l2(l2)
    inp = layers.Input(shape=(L, 3))
    x = inp
    for b in range(hl):
        x = layers.GRU(units, return_sequences=(b < hl - 1), dropout=dropout, kernel_regularizer=reg)(x)
    out = layers.Dense(2, activation="softmax")(x)
    return tf.keras.Model(inp, out, name=f"gru_L{L}_HL{hl}")


class KerasPredictor:
    backend = "keras"

    def __init__(self, L, hl, arch="cnn", epochs=60, batch=256, lr=1e-3, patience=10, l2=1e-3, seed=0, verbose=2,
                 class_weight=None):
        self.class_weight = class_weight
        self.L, self.hl, self.arch = L, hl, arch
        self.epochs, self.batch, self.lr, self.patience, self.l2, self.seed, self.verbose = \
            epochs, batch, lr, patience, l2, seed, verbose
        self.model = None

    def fit(self, Xtr, ytr, Xva, yva):
        tf = _tf(); tf.keras.utils.set_random_seed(self.seed)
        build = build_resnet1d if self.arch == "cnn" else build_gru
        self.model = build(self.L, self.hl, l2=self.l2)
        self.model.compile(tf.keras.optimizers.Adam(self.lr),
                           loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.patience, restore_best_weights=True),
              tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5)]
        w = resolve_class_weight(self.class_weight, ytr)
        cwd = None if w is None else {0: float(w[0]), 1: float(w[1])}
        hist = self.model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=self.epochs,
                              batch_size=self.batch, callbacks=cb, verbose=self.verbose, class_weight=cwd)
        return hist.history

    def predict(self, X):
        p = self.model.predict(X, batch_size=4096, verbose=0)
        return p.argmax(1).astype(np.int8)

    def predict_proba(self, X):
        p = self.model.predict(X, batch_size=4096, verbose=0)
        return p.astype(np.float32)

    def n_params(self):
        return int(self.model.count_params())

    def exec_ms(self, n=100):
        x = np.zeros((1, self.L, 3), np.float32)
        f = self.model
        f(x, training=False)
        t = time.perf_counter()
        for _ in range(n): f(x, training=False)
        return (time.perf_counter() - t) / n * 1e3

    def save(self, stem):
        path = stem if stem.endswith(".keras") else stem + ".keras"
        self.model.save(path)

    @classmethod
    def load(cls, path):
        tf = _tf()
        m = tf.keras.models.load_model(path)
        L = m.input_shape[1]
        o = cls(L, hl=0); o.model = m
        return o


# ===================================================================== sklearn surrogate
def window_features(X):
    """(N, L, 3) normalised windows -> (N, F) hand-crafted features."""
    V, I, C = X[..., 0], X[..., 1], X[..., 2]
    dV = np.diff(V, axis=1) if X.shape[1] > 1 else np.zeros((len(X), 1), dtype=np.float32)
    dI = np.diff(I, axis=1) if X.shape[1] > 1 else np.zeros((len(X), 1), dtype=np.float32)

    f = [V.mean(1), V.std(1), V.min(1), V.max(1), V[:, -1] - V[:, 0],
         I.mean(1), I.std(1), I.min(1), I.max(1),
         C.mean(1), C[:, -1] - C[:, 0],
         dV.std(1), np.abs(dV).mean(1), np.abs(dV).max(1), dV.mean(1),
         dI.std(1), np.abs(dI).mean(1),
         (np.sign(dV[:, 1:]) != np.sign(dV[:, :-1])).mean(1) if dV.shape[1] > 1 else np.zeros(len(X))]
    return np.stack(f, 1).astype(np.float32)


class SklearnPredictor:
    backend = "sklearn"
    _CAP = {1: dict(max_iter=60, max_depth=3), 2: dict(max_iter=150, max_depth=5), 3: dict(max_iter=300, max_depth=7)}

    def __init__(self, L, hl, seed=0, class_weight=None, **_):
        self.L, self.hl, self.seed = L, hl, seed
        self.class_weight = class_weight
        self.clf = None

    def fit(self, Xtr, ytr, Xva=None, yva=None):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.clf = HistGradientBoostingClassifier(learning_rate=0.1, random_state=self.seed, **self._CAP.get(self.hl, self._CAP[1]))
        w = resolve_class_weight(self.class_weight, ytr)
        sw = None if w is None else w[np.asarray(ytr).astype(int)]
        self.clf.fit(window_features(Xtr), ytr, sample_weight=sw)
        return {}

    def predict(self, X):
        return self.clf.predict(window_features(X)).astype(np.int8)

    def predict_proba(self, X):
        return self.clf.predict_proba(window_features(X)).astype(np.float32)

    def n_params(self):
        return int(sum(len(t[0].nodes) for t in self.clf._predictors))

    def exec_ms(self, n=200):
        x = np.zeros((1, self.L, 3), np.float32)
        self.predict(x)
        t = time.perf_counter()
        for _ in range(n): self.predict(x)
        return (time.perf_counter() - t) / n * 1e3

    def save(self, stem):
        import joblib
        path = stem if stem.endswith(".joblib") else stem + ".joblib"
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        import joblib
        return joblib.load(path)


# ===================================================================== Factory & Loader
def make_predictor(backend, L, hl, arch="cnn", **kw):
    if backend == "torch":   return TorchPredictor(L, hl, arch=arch, **kw)
    if backend == "keras":   return KerasPredictor(L, hl, arch=arch, **kw)
    if backend == "sklearn": return SklearnPredictor(L, hl, **kw)
    raise ValueError(f"Unknown backend: {backend}. Choose 'torch', 'keras', or 'sklearn'.")


def load_predictor(stem_or_path):
    """Loads a saved predictor model across PyTorch, Sklearn, or Keras formats."""
    # Check if exact file path exists
    if os.path.exists(stem_or_path):
        if stem_or_path.endswith(".pt"):
            return TorchPredictor.load(stem_or_path)
        elif stem_or_path.endswith(".joblib"):
            return SklearnPredictor.load(stem_or_path)
        elif stem_or_path.endswith(".keras"):
            return KerasPredictor.load(stem_or_path)

    # Search for matching extension
    for ext, cls in ((".pt", TorchPredictor), (".joblib", SklearnPredictor), (".keras", KerasPredictor)):
        p = stem_or_path + ext
        if os.path.exists(p):
            return cls.load(p)

    raise FileNotFoundError(f"No model found for stem or path: {stem_or_path} (checked .pt, .joblib, .keras)")
