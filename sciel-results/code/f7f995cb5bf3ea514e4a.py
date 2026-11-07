"""Improved symbolic regression with rich basis and ridge regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None
TIME_LIMIT = 295.0


def _time_left():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        return TIME_LIMIT - (time.time() - _START_TIME)
    return TIME_LIMIT - (time.time() - _START_TIME)


def _build_features(X):
    """Build a rich set of basis functions for 1D input."""
    n = len(X)
    d = X.shape[1]
    cols = [np.ones(n)]

    if d == 1:
        x = X[:, 0]
        # Polynomial terms up to degree 8
        for k in range(1, 9):
            cols.append(x ** k)
        # Trigonometric terms
        cols.append(np.sin(x))
        cols.append(np.cos(x))
        cols.append(np.sin(2 * x))
        cols.append(np.cos(2 * x))
        cols.append(np.sin(3 * x))
        cols.append(np.cos(3 * x))
        cols.append(np.sin(4 * x))
        cols.append(np.cos(4 * x))
        # Exponential terms
        cols.append(np.exp(x))
        cols.append(np.exp(-x))
        # Hyperbolic
        cols.append(np.tanh(x))
        # Rational functions
        cols.append(1.0 / (1.0 + x ** 2))
        cols.append(1.0 / (1.0 + x ** 4))
        # Log terms (with safe handling)
        abs_x = np.abs(x)
        cols.append(np.log(1.0 + abs_x))
        cols.append(np.log1p(abs_x))
        # Sqrt terms
        cols.append(np.sqrt(abs_x))
        # Reciprocal (with safe handling)
        safe_x = np.where(np.abs(x) > 1e-6, x, np.sign(x) * 1e-6)
        cols.append(1.0 / safe_x)
        # x * sin(x), x * cos(x)
        cols.append(x * np.sin(x))
        cols.append(x * np.cos(x))
        # x^2 * sin(x), x^2 * cos(x)
        cols.append(x ** 2 * np.sin(x))
        cols.append(x ** 2 * np.cos(x))
    else:
        # For multi-dimensional (shouldn't happen per task, but safe)
        for j in range(d):
            xj = X[:, j]
            for k in range(1, 5):
                cols.append(xj ** k)
            cols.append(np.sin(xj))
            cols.append(np.cos(xj))
            cols.append(np.exp(xj))
            cols.append(np.exp(-xj))
            cols.append(np.tanh(xj))
            cols.append(1.0 / (1.0 + xj ** 2))

    return np.column_stack(cols)


def fit(X_train, y_train):
    """Fit a rich basis expansion using ridge regression."""
    if _time_left() < 5.0:
        # Fallback: simple linear regression
        X1 = np.column_stack([np.ones(len(X_train)), X_train])
        w, *_ = np.linalg.lstsq(X1, y_train, rcond=None)
        return lambda X: np.column_stack([np.ones(len(X)), X]) @ w

    X1 = _build_features(X_train)
    y1 = np.asarray(y_train, dtype=np.float64)

    # Standardize features for better conditioning
    mu = np.mean(X1, axis=0)
    sigma = np.std(X1, axis=0)
    sigma[sigma < 1e-8] = 1.0
    X1_std = (X1 - mu) / sigma

    # Try ridge regression with multiple alpha values
    best_w = None
    best_r2 = -np.inf

    for alpha in [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0]:
        try:
            # Regularized least squares: (X^T X + alpha*I) w = X^T y
            XtX = X1_std.T @ X1_std
            Xty = X1_std.T @ y1
            reg = XtX + alpha * np.eye(XtX.shape[0])
            w_std = np.linalg.solve(reg, Xty)

            # Check for NaN/Inf
            if not np.all(np.isfinite(w_std)):
                continue

            # Compute R^2 on training data
            pred = X1_std @ w_std
            ss_res = np.sum((y1 - pred) ** 2)
            ss_tot = np.sum((y1 - np.mean(y1)) ** 2)
            if ss_tot < 1e-10:
                r2 = 0.0
            else:
                r2 = 1.0 - ss_res / ss_tot

            if r2 > best_r2:
                best_r2 = r2
                best_w = w_std
        except Exception:
            continue

    if best_w is None:
        # Fallback to OLS
        w_std, *_ = np.linalg.lstsq(X1_std, y1, rcond=None)

    w = best_w

    def predict(X):
        if _time_left() < 1.0:
            # Emergency: return mean
            return np.full(len(X), np.mean(y_train))
        Xf = _build_features(X)
        Xf_std = (Xf - mu) / sigma
        return Xf_std @ w

    return predict
# EVOLVE-BLOCK-END
