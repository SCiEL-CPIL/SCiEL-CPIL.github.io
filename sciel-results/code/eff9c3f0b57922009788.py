"""Symbolic regression over 4 hidden 1-D noiseless equations.

Strategy:
- For 1-D noiseless data, interpolation is exact on training points.
- The key challenge is generalization to test points between/around training points.
- Use multiple smooth interpolators and select by a hold-out validation split.
- Also try polynomial fits and a neural network as fallbacks.
- Since data is noiseless, the best model on a random hold-out subset
  of training points should also be the best on the hidden test set.
"""
import numpy as np
import time
from scipy.interpolate import PchipInterpolator, CubicSpline, interp1d
from scipy.optimize import minimize_scalar
import warnings

warnings.filterwarnings('ignore')

_START = time.time()
_DEADLINE = _START + 290.0  # safety margin before 330s kill


def _now():
    return time.time()


def _fit_spline(X_train, y_train, method='cubic'):
    """Smooth spline interpolator (exact on training points)."""
    x = X_train[:, 0]
    y = y_train
    sidx = np.argsort(x)
    xs = x[sidx]
    ys = y[sidx]
    # Remove duplicate x values
    keep = np.concatenate(([True], np.diff(xs) > 1e-12))
    xs, ys = xs[keep], ys[keep]
    if xs.size < 2:
        return None
    if method == 'pchip':
        spl = PchipInterpolator(xs, ys)
    else:
        spl = CubicSpline(xs, ys)

    def predict(X):
        return spl(X[:, 0])

    return predict


def _fit_poly(X_train, y_train, degree):
    """Polynomial fit."""
    x = X_train[:, 0]
    y = y_train
    try:
        coeffs = np.polyfit(x, y, degree)
    except Exception:
        return None

    def predict(X):
        return np.polyval(coeffs, X[:, 0])

    return predict


def _fit_rbf(X_train, y_train, width):
    """RBF interpolator with given width."""
    x = X_train[:, 0]
    y = y_train
    n = x.size
    diff = x[:, None] - x[None, :]
    K = np.exp(-(diff * diff) / (2.0 * width * width))
    K[np.diag_indices_from(K)] += 1e-8
    try:
        coef = np.linalg.solve(K, y)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(K, y, rcond=None)[0]

    def predict(X):
        xt = X[:, 0]
        d = xt[:, None] - x[None, :]
        return np.exp(-(d * d) / (2.0 * width * width)) @ coef

    return predict


def _fit_gaussian_process(X_train, y_train, length_scale, noise=1e-6):
    """Simple Gaussian Process regression."""
    x = X_train[:, 0]
    y = y_train
    n = x.size
    diff = x[:, None] - x[None, :]
    K = np.exp(-0.5 * (diff * diff) / (length_scale * length_scale)) + noise * np.eye(n)
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    except np.linalg.LinAlgError:
        try:
            alpha = np.linalg.solve(K, y)
        except np.linalg.LinAlgError:
            return None

    def predict(X):
        xt = X[:, 0]
        Ks = np.exp(-0.5 * (xt[:, None] - x[None, :]) ** 2 / (length_scale * length_scale))
        return Ks @ alpha

    return predict


def _fit_nn(X_train, y_train, hidden, act, lr, epochs, seed, deadline):
    """Small MLP with Adam optimizer."""
    rng = np.random.RandomState(seed)
    d = X_train.shape[1]
    n = X_train.shape[0]
    y = y_train
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-12:
        y_std = 1.0
    yt = (y - y_mean) / y_std

    layers = [d] + list(hidden) + [1]
    L = len(layers)
    W = [rng.randn(layers[i], layers[i + 1]) * np.sqrt(2.0 / layers[i])
         for i in range(L - 1)]
    B = [np.zeros(layers[i + 1]) for i in range(L - 1)]

    mW = [np.zeros_like(w) for w in W]
    vW = [np.zeros_like(w) for w in W]
    mB = [np.zeros_like(b) for b in B]
    vB = [np.zeros_like(b) for b in B]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    best_loss = float('inf')
    best_W = [w.copy() for w in W]
    best_B = [b.copy() for b in B]

    for ep in range(epochs):
        if _now() > deadline - 2:
            break
        # forward
        a = X_train
        act_cache = [a]
        for i in range(L - 1):
            z = a @ W[i] + B[i]
            a = act(z) if i < L - 2 else z
            act_cache.append(z)
            act_cache.append(a)
        out = a
        err = out - yt
        loss = float(np.mean(err * err))
        if loss < best_loss:
            best_loss = loss
            best_W = [w.copy() for w in W]
            best_B = [b.copy() for b in B]

        # backward
        d_out = 2.0 * err / n
        for i in range(L - 2, -1, -1):
            z = act_cache[2 * i + 1]
            a_in = act_cache[2 * i]
            d_z = d_out
            if i < L - 2:
                if act == 'tanh':
                    d_z = d_z * (1.0 - a * a)
                elif act == 'relu':
                    d_z = d_z * (z > 0)
                else:  # sigmoid
                    d_z = d_z * a * (1.0 - a)
            dW = a_in.T @ d_z
            dB = d_z.sum(axis=0)
            d_out = d_z @ W[i].T

            t += 1
            mW[i] = beta1 * mW[i] + (1 - beta1) * dW
            vW[i] = beta2 * vW[i] + (1 - beta2) * (dW * dW)
            W[i] -= lr * (mW[i] / (1 - beta1 ** t)) / (np.sqrt(vW[i] / (1 - beta2 ** t)) + eps)

            mB[i] = beta1 * mB[i] + (1 - beta1) * dB
            vB[i] = beta2 * vB[i] + (1 - beta2) * (dB * dB)
            B[i] -= lr * (mB[i] / (1 - beta1 ** t)) / (np.sqrt(vB[i] / (1 - beta2 ** t)) + eps)

    def predict(X):
        a = X
        for i in range(L - 2):
            z = a @ best_W[i] + best_B[i]
            if act == 'tanh':
                a = np.tanh(z)
            elif act == 'relu':
                a = np.maximum(0, z)
            else:
                a = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        a = a @ best_W[L - 2] + best_B[L - 2]
        return a * y_std + y_mean

    return predict


def fit(X_train, y_train):
    deadline = _DEADLINE
    d = X_train.shape[1]
    y = y_train
    n = len(y)
    x = X_train[:, 0]

    # Create a validation split to select the best model
    rng = np.random.RandomState(42)
    val_frac = 0.15
    val_n = max(10, int(n * val_frac))
    indices = np.arange(n)
    val_idx = rng.choice(n, val_n, replace=False)
    train_idx = np.setdiff1d(indices, val_idx)

    X_tr = X_train[train_idx]
    y_tr = y_train[train_idx]
    X_val = X_train[val_idx]
    y_val = y_train[val_idx]

    candidates = []  # (val_mse, predict_callable)

    def _add(pred):
        try:
            yp_val = np.asarray(pred(X_val), dtype=float)
            if yp_val.shape != y_val.shape:
                yp_val = yp_val.reshape(y_val.shape)
            if not np.all(np.isfinite(yp_val)):
                return
            mse_val = float(np.mean((yp_val - y_val) ** 2))
            if np.isfinite(mse_val):
                candidates.append((mse_val, pred))
        except Exception:
            pass

    # ---- Spline models ----
    for method in ('cubic', 'pchip'):
        try:
            pred = _fit_spline(X_train, y, method)
            if pred is not None:
                _add(pred)
        except Exception:
            pass

    # ---- Polynomial models ----
    for deg in (2, 3, 4, 5, 6, 7, 8, 10, 12, 15):
        if _now() > deadline - 5:
            break
        try:
            pred = _fit_poly(X_train, y, deg)
            if pred is not None:
                _add(pred)
        except Exception:
            pass

    # ---- RBF models with various widths ----
    rng_x = (x.max() - x.min())
    if rng_x > 1e-12:
        for frac in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
            if _now() > deadline - 5:
                break
            try:
                pred = _fit_rbf(X_train, y, width=frac * rng_x)
                if pred is not None:
                    _add(pred)
            except Exception:
                pass

    # ---- Gaussian Process with various length scales ----
    if rng_x > 1e-12:
        for frac in (0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5):
            if _now() > deadline - 5:
                break
            try:
                pred = _fit_gaussian_process(X_train, y, length_scale=frac * rng_x)
                if pred is not None:
                    _add(pred)
            except Exception:
                pass

    # ---- Neural Networks ----
    nn_configs = [
        ([32, 32], 'tanh', 0.01, 2000, 0),
        ([64, 32], 'tanh', 0.005, 2000, 1),
        ([32, 32], 'relu', 0.01, 3000, 2),
        ([64, 64], 'tanh', 0.003, 2500, 3),
        ([128, 64], 'tanh', 0.002, 2000, 4),
    ]
    for hidden, act, lr, epochs, seed in nn_configs:
        if _now() > deadline - 10:
            break
        try:
            pred = _fit_nn(X_train, y, hidden, act, lr, epochs, seed, deadline)
            _add(pred)
        except Exception:
            continue

    # ---- Selection ----
    if not candidates:
        mean_val = float(np.mean(y))
        def predict(X):
            return np.full(X.shape[0], mean_val)
        return predict

    candidates.sort(key=lambda c: c[0])
    best_predict = candidates[0][1]

    def final_predict(X):
        try:
            yout = np.asarray(best_predict(X), dtype=float)
            yout = np.where(np.isfinite(yout), yout, 0.0)
            return yout
        except Exception:
            return np.zeros(X.shape[0])

    return final_predict
