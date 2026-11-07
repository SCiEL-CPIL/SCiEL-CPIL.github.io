"""Symbolic regression via ensemble of diverse models with time budget guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d

# Global deadline: 330s wall-clock, use 310s as safety margin
_DEADLINE = time.time() + 310.0


def _remaining():
    return _DEADLINE - time.time()


def _poly_fit(X, y, degree):
    """Polynomial fit of given degree."""
    n = len(X)
    # Build Vandermonde-like feature matrix for 1D case
    x = X[:, 0]
    cols = [x ** k for k in range(degree + 1)]
    F = np.column_stack(cols) if degree > 0 else np.ones((n, 1))
    w, *_ = np.linalg.lstsq(F, y, rcond=None)
    return lambda X: (np.column_stack([X[:, 0] ** k for k in range(degree + 1)]) if degree > 0 else np.ones((len(X), 1))) @ w


def _spline_fit(X, y, n_knots):
    """Interpolating spline through sorted data points."""
    x = X[:, 0]
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    # Use cubic interpolation if enough points, else linear
    if len(x_sorted) >= 4:
        f = interp1d(x_sorted, y_sorted, kind='cubic', fill_value='extrapolate')
    else:
        f = interp1d(x_sorted, y_sorted, kind='linear', fill_value='extrapolate')
    return lambda X: f(X[:, 0])


def _gaussian_process_fit(X, y, length_scale, variance):
    """Simple GP with RBF kernel."""
    x = X[:, 0]
    n = len(x)
    
    def kernel(a, b):
        d = (a[:, None] - b[None, :]) ** 2
        return variance * np.exp(-0.5 * d / (length_scale ** 2))
    
    K = kernel(x, x) + 1e-8 * np.eye(n)
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    except np.linalg.LinAlgError:
        return None
    
    def predict(X_new):
        x_new = X_new[:, 0]
        K_star = kernel(x_new, x)
        return K_star @ alpha
    return predict


def _neural_net_fit(X, y, hidden_size, n_epochs=500, lr=0.01):
    """Simple MLP with one hidden layer, trained via gradient descent."""
    x = X[:, 0].reshape(-1, 1)
    y_col = y.reshape(-1, 1)
    n = len(x)
    
    # Initialize weights
    np.random.seed(42)
    W1 = np.random.randn(1, hidden_size) * np.sqrt(2.0 / 1)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, 1) * np.sqrt(2.0 / hidden_size)
    b2 = np.zeros((1, 1))
    
    for epoch in range(n_epochs):
        if _remaining() < 5:
            break
        # Forward
        z1 = x @ W1 + b1
        a1 = np.tanh(z1)
        z2 = a1 @ W2 + b2
        pred = z2
        
        # Backward
        dz2 = (2.0 / n) * (pred - y_col)
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)
        da1 = dz2 @ W2.T
        dz1 = da1 * (1 - a1 ** 2)
        dW1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # Update
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
    
    def predict(X_new):
        x_new = X_new[:, 0].reshape(-1, 1)
        z1 = x_new @ W1 + b1
        a1 = np.tanh(z1)
        z2 = a1 @ W2 + b2
        return z2.flatten()
    return predict


def _fourier_fit(X, y, n_terms):
    """Fourier series fit."""
    x = X[:, 0]
    cols = [np.ones_like(x)]
    for k in range(1, n_terms + 1):
        cols.append(np.sin(k * x))
        cols.append(np.cos(k * x))
    F = np.column_stack(cols)
    w, *_ = np.linalg.lstsq(F, y, rcond=None)
    
    def predict(X_new):
        x_new = X_new[:, 0]
        cols = [np.ones_like(x_new)]
        for k in range(1, n_terms + 1):
            cols.append(np.sin(k * x_new))
            cols.append(np.cos(k * x_new))
        F_new = np.column_stack(cols)
        return F_new @ w
    return predict


def fit(X_train, y_train):
    """Fit multiple models and return the best one based on training MSE."""
    X = X_train.astype(np.float64)
    y = y_train.astype(np.float64)
    
    candidates = []
    
    # 1. Polynomial fits of various degrees
    for deg in [1, 2, 3, 4, 5, 6, 8, 10]:
        if _remaining() < 10:
            break
        try:
            pred_fn = _poly_fit(X, y, deg)
            train_pred = pred_fn(X)
            mse = np.mean((train_pred - y) ** 2)
            candidates.append((mse, pred_fn))
        except Exception:
            pass
    
    # 2. Spline interpolation
    if _remaining() > 10:
        try:
            pred_fn = _spline_fit(X, y, 10)
            train_pred = pred_fn(X)
            mse = np.mean((train_pred - y) ** 2)
            candidates.append((mse, pred_fn))
        except Exception:
            pass
    
    # 3. Fourier series
    for n_terms in [3, 5, 8, 12, 20]:
        if _remaining() < 10:
            break
        try:
            pred_fn = _fourier_fit(X, y, n_terms)
            train_pred = pred_fn(X)
            mse = np.mean((train_pred - y) ** 2)
            candidates.append((mse, pred_fn))
        except Exception:
            pass
    
    # 4. Neural network
    if _remaining() > 20:
        for hidden in [8, 16, 32]:
            if _remaining() < 15:
                break
            try:
                pred_fn = _neural_net_fit(X, y, hidden, n_epochs=300, lr=0.01)
                train_pred = pred_fn(X)
                mse = np.mean((train_pred - y) ** 2)
                candidates.append((mse, pred_fn))
            except Exception:
                pass
    
    # 5. Gaussian Process with a few hyperparameter settings
    if _remaining() > 15:
        var_y = np.var(y) + 1e-10
        for ls in [0.1, 0.3, 0.5, 1.0, 2.0]:
            if _remaining() < 10:
                break
            try:
                pred_fn = _gaussian_process_fit(X, y, ls, var_y)
                if pred_fn is not None:
                    train_pred = pred_fn(X)
                    mse = np.mean((train_pred - y) ** 2)
                    candidates.append((mse, pred_fn))
            except Exception:
                pass
    
    # Select best model by training MSE
    if not candidates:
        # Fallback: linear fit
        pred_fn = _poly_fit(X, y, 1)
        return pred_fn
    
    candidates.sort(key=lambda t: t[0])
    best_mse, best_pred = candidates[0]
    
    # Final safety check
    def safe_predict(X_new):
        try:
            result = best_pred(X_new)
            result = np.asarray(result, dtype=np.float64).flatten()
            # Ensure finite values
            if not np.all(np.isfinite(result)):
                result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
            return result
        except Exception:
            return np.zeros(len(X_new))
    
    return safe_predict
# EVOLVE-BLOCK-END
