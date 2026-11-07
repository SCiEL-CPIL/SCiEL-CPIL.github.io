"""Symbolic regression via rich basis + ridge regression with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_TIME_LIMIT = 300.0
_SAFETY = 15.0
_START = time.time()


def _time_left():
    return _TIME_LIMIT - _SAFETY - (time.time() - _START)


def _build_rich_features(x):
    """x: 1D array. Returns a 2D array of basis functions (n, p)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if n == 0:
        return np.empty((0, 0))

    cols = []
    # Polynomial terms
    for k in range(1, 8):
        cols.append(x ** k)
    # Trigonometric
    cols.append(np.sin(x))
    cols.append(np.cos(x))
    cols.append(np.sin(2.0 * x))
    cols.append(np.cos(2.0 * x))
    cols.append(np.sin(3.0 * x))
    cols.append(np.cos(3.0 * x))
    # Exponential (clipped for safety)
    x_safe = np.clip(x, -20, 20)
    cols.append(np.exp(x_safe))
    cols.append(np.exp(-x_safe))
    # Hyperbolic
    cols.append(np.tanh(x))
    cols.append(np.arctan(x))
    # Rational-ish / smooth
    cols.append(1.0 / (1.0 + x ** 2))
    cols.append(np.log1p(x ** 2))
    # Square root of abs
    cols.append(np.sqrt(np.abs(x)))
    # Absolute value
    cols.append(np.abs(x))

    A = np.column_stack(cols)
    return A


def _scale_features(A):
    """Standardize each column to zero mean, unit variance (with guard)."""
    n, p = A.shape
    if p == 0:
        return A, np.zeros(0), np.ones(0)
    mu = A.mean(axis=0)
    sd = A.std(axis=0)
    sd[sd < 1e-12] = 1.0
    A_scaled = (A - mu) / sd
    return A_scaled, mu, sd


def _ridge_solve(A_scaled, y, lam):
    """Solve (A^T A + lam I) w = A^T y via Cholesky or lstsq fallback."""
    n, p = A_scaled.shape
    if p == 0:
        return np.zeros(0)
    try:
        G = A_scaled.T @ A_scaled + lam * np.eye(p)
        b = A_scaled.T @ y
        # Use Cholesky for speed and stability
        L = np.linalg.cholesky(G)
        w = np.linalg.solve(np.linalg.inv(L).T, np.linalg.solve(L, b))
        return w
    except np.linalg.LinAlgError:
        # Fallback to lstsq
        w, *_ = np.linalg.lstsq(A_scaled, y, rcond=None)
        return w


def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).ravel()
    n, d = X_train.shape

    # Extract the single input column (task says 1 input)
    if d >= 1:
        x_train = X_train[:, 0]
    else:
        x_train = X_train.ravel()

    # Build features
    A = _build_rich_features(x_train)
    A_scaled, mu, sd = _scale_features(A)

    # Try a few regularization strengths, pick best via training MSE
    # (noiseless, so training MSE is a good proxy; regularization guards against
    #  collinearity in the rich basis)
    best_w = None
    best_mse = np.inf
    lams = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

    for lam in lams:
        if _time_left() < 5.0:
            break
        w = _ridge_solve(A_scaled, y_train, lam)
        pred = A_scaled @ w
        mse = np.mean((pred - y_train) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_w = w

    if best_w is None:
        # Fallback: simple linear fit
        w = np.linalg.lstsq(A_scaled, y_train, rcond=None)[0]
        best_w = w

    w_final = best_w
    mu_final = mu
    sd_final = sd

    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] >= 1:
            x = X[:, 0]
        else:
            x = X.ravel()
        A = _build_rich_features(x)
        A_scaled = (A - mu_final) / sd_final
        return A_scaled @ w_final

    return predict
# EVOLVE-BLOCK-END
