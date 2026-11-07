"""Symbolic regression over 10 hidden equations with rich basis and forward selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 310  # safety margin before 330s wall-clock kill


def _time_left():
    return _DEADLINE - time.time()


def _build_rich_basis(X, max_poly_degree=3):
    """Build a rich feature matrix including polynomial, trig, exp, log, tanh terms."""
    d = X.shape[1]
    n = X.shape[0]
    cols = [np.ones(n)]
    
    # Polynomial terms up to max_poly_degree
    if max_poly_degree >= 1:
        for j in range(d):
            cols.append(X[:, j])
    
    if max_poly_degree >= 2:
        for i in range(d):
            for j in range(i, d):
                cols.append(X[:, i] * X[:, j])
    
    if max_poly_degree >= 3:
        for i in range(d):
            for j in range(i, d):
                for k in range(j, d):
                    cols.append(X[:, i] * X[:, j] * X[:, k])
    
    # Trigonometric terms (sin, cos) of individual inputs
    for j in range(d):
        cols.append(np.sin(X[:, j]))
        cols.append(np.cos(X[:, j]))
    
    # tanh of individual inputs
    for j in range(d):
        cols.append(np.tanh(X[:, j]))
    
    # log of absolute values (safe version)
    for j in range(d):
        cols.append(np.log(np.abs(X[:, j]) + 1e-8))
    
    # exp of scaled inputs (to avoid overflow)
    for j in range(d):
        scaled = 0.5 * X[:, j]
        cols.append(np.exp(np.clip(scaled, -10, 10)))
    
    # Pairwise products of trig/tanh with linear
    for j in range(d):
        cols.append(X[:, j] * np.sin(X[:, j]))
        cols.append(X[:, j] * np.cos(X[:, j]))
        cols.append(X[:, j] * np.tanh(X[:, j]))
    
    # Pairwise polynomial products with trig
    if d >= 2:
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(X[:, i] * np.sin(X[:, j]))
                cols.append(X[:, i] * np.cos(X[:, j]))
                cols.append(X[:, i] * np.tanh(X[:, j]))
    
    return np.column_stack(cols)


def _fit_ridge(F, y, alpha):
    """Fit ridge regression and return weights and training MSE."""
    n_features = F.shape[1]
    F_aug = np.hstack([F, alpha * np.eye(n_features)])
    y_aug = np.concatenate([y, np.zeros(n_features)])
    w, *_ = np.linalg.lstsq(F_aug, y_aug, rcond=None)
    pred = F @ w
    mse = np.mean((pred - y) ** 2)
    return w, mse


def _forward_select(F, y, max_features=30, max_time=25):
    """Forward feature selection to pick the most useful features."""
    n_samples, n_features = F.shape
    selected = []
    remaining = list(range(n_features))
    
    # Always start with intercept (index 0)
    selected.append(0)
    remaining.remove(0)
    
    best_mse = np.inf
    start_time = time.time()
    
    for _ in range(min(max_features, len(remaining))):
        if time.time() - start_time > max_time:
            break
        
        best_j = None
        best_mse_iter = best_mse
        
        for j in remaining:
            candidate = selected + [j]
            F_sub = F[:, candidate]
            w, mse = _fit_ridge(F_sub, y, alpha=1e-4)
            if mse < best_mse_iter:
                best_mse_iter = mse
                best_j = j
        
        if best_j is None:
            break
        
        selected.append(best_j)
        remaining.remove(best_j)
        best_mse = best_mse_iter
    
    F_sub = F[:, selected]
    w, mse = _fit_ridge(F_sub, y, alpha=1e-4)
    return selected, w, mse


def fit(X_train, y_train):
    d = X_train.shape[1]
    
    # Build rich basis
    F = _build_rich_basis(X_train, max_poly_degree=3)
    
    # Handle potential NaN/Inf
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Forward selection with time budget
    max_time_per_eq = 20  # seconds per equation for selection
    selected, w, mse = _forward_select(F, y_train, max_features=25, max_time=max_time_per_eq)
    
    # If forward selection failed, fall back to full ridge
    if len(selected) < 2:
        w, mse = _fit_ridge(F, y_train, alpha=1e-4)
        selected = list(range(F.shape[1]))
    
    w_final = w
    selected_final = selected
    
    def predict(X):
        F_pred = _build_rich_basis(X, max_poly_degree=3)
        F_pred = np.nan_to_num(F_pred, nan=0.0, posinf=0.0, neginf=0.0)
        F_sub = F_pred[:, selected_final]
        return F_sub @ w_final
    
    return predict
# EVOLVE-BLOCK-END
