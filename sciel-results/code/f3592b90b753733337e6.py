"""Symbolic regression via SINDy-style sparse identification over a rich basis."""
import numpy as np
import time
import itertools
from scipy.linalg import lstsq
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lsmr
import warnings
warnings.filterwarnings('ignore')


def _build_library(X):
    """
    Build a library of candidate basis functions from X.
    X: (n, d)
    Returns: (n, m) matrix of basis evaluations, and list of basis names.
    """
    n, d = X.shape
    cols = [np.ones(n)]
    names = ['1']
    
    # Linear terms
    for j in range(d):
        cols.append(X[:, j])
        names.append(f'x{j}')
    
    # Quadratic terms
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
            names.append(f'x{i}x{j}')
    
    # Cubic terms
    for i in range(d):
        for j in range(i, d):
            for k in range(j, d):
                cols.append(X[:, i] * X[:, j] * X[:, k])
                names.append(f'x{i}x{j}x{k}')
    
    # Trig terms
    for j in range(d):
        cols.append(np.sin(X[:, j]))
        names.append(f'sin(x{j})')
        cols.append(np.cos(X[:, j]))
        names.append(f'cos(x{j})')
        cols.append(np.sin(2 * X[:, j]))
        names.append(f'sin(2x{j})')
        cols.append(np.cos(2 * X[:, j]))
        names.append(f'cos(2x{j})')
    
    # Exponential terms (careful with overflow)
    for j in range(d):
        # Scale to avoid overflow: use exp(x/10) etc.
        for scale in [0.1, 0.5, 1.0]:
            vals = np.exp(scale * X[:, j])
            # Clip to avoid inf
            vals = np.clip(vals, -1e10, 1e10)
            cols.append(vals)
            names.append(f'exp({scale}x{j})')
    
    # Log terms
    for j in range(d):
        # Ensure positive for log
        xj = np.maximum(X[:, j], 1e-10)
        cols.append(np.log(xj))
        names.append(f'log(x{j})')
        cols.append(np.log(xj + 1))
        names.append(f'log(x{j}+1)')
    
    # Reciprocal terms
    for j in range(d):
        xj = np.maximum(X[:, j], 1e-10)
        cols.append(1.0 / xj)
        names.append(f'1/x{j}')
    
    # Sqrt terms
    for j in range(d):
        xj = np.maximum(X[:, j], 1e-10)
        cols.append(np.sqrt(xj))
        names.append(f'sqrt(x{j})')
    
    # Cross-term interactions with trig
    for i in range(d):
        for j in range(i+1, d):
            cols.append(X[:, i] * np.sin(X[:, j]))
            names.append(f'x{i}sin(x{j})')
            cols.append(X[:, i] * np.cos(X[:, j]))
            names.append(f'x{i}cos(x{j})')
    
    # Product of trig
    for i in range(d):
        for j in range(i+1, d):
            cols.append(np.sin(X[:, i]) * np.cos(X[:, j]))
            names.append(f'sin(x{i})cos(x{j})')
            cols.append(np.cos(X[:, i]) * np.sin(X[:, j]))
            names.append(f'cos(x{i})sin(x{j})')
    
    # Higher-order mixed
    for i in range(d):
        for j in range(i, d):
            for k in range(j, d):
                cols.append(X[:, i] * np.sin(X[:, j]) * np.cos(X[:, k]))
                names.append(f'x{i}sin(x{j})cos(x{k})')
    
    # Absolute value
    for j in range(d):
        cols.append(np.abs(X[:, j]))
        names.append(f'|x{j}|')
    
    # Squared trig
    for j in range(d):
        cols.append(np.sin(X[:, j])**2)
        names.append(f'sin^2(x{j})')
        cols.append(np.cos(X[:, j])**2)
        names.append(f'cos^2(x{j})')
    
    # Tanh
    for j in range(d):
        cols.append(np.tanh(X[:, j]))
        names.append(f'tanh(x{j})')
    
    # Rational: x_i / x_j
    for i in range(d):
        for j in range(d):
            if i != j:
                denom = np.maximum(X[:, j], 1e-10)
                cols.append(X[:, i] / denom)
                names.append(f'x{i}/x{j}')
    
    # x_i^2 / x_j
    for i in range(d):
        for j in range(d):
            if i != j:
                denom = np.maximum(X[:, j], 1e-10)
                cols.append(X[:, i]**2 / denom)
                names.append(f'x{i}^2/x{j}')
    
    # x_i * x_j / x_k
    for i in range(d):
        for j in range(i, d):
            for k in range(d):
                if k != i and k != j:
                    denom = np.maximum(X[:, k], 1e-10)
                    cols.append(X[:, i] * X[:, j] / denom)
                    names.append(f'x{i}x{j}/x{k}')
    
    # exp(-x)
    for j in range(d):
        cols.append(np.exp(-X[:, j]))
        names.append(f'exp(-x{j})')
        cols.append(np.exp(-0.5 * X[:, j]))
        names.append(f'exp(-0.5x{j})')
    
    # sinh, cosh
    for j in range(d):
        cols.append(np.sinh(0.1 * X[:, j]))
        names.append(f'sinh(0.1x{j})')
        cols.append(np.cosh(0.1 * X[:, j]))
        names.append(f'cosh(0.1x{j})')
    
    # Power terms
    for j in range(d):
        xj = np.maximum(X[:, j], 1e-10)
        cols.append(xj**0.5)
        names.append(f'x{j}^0.5')
        cols.append(xj**2.0)
        names.append(f'x{j}^2')
        cols.append(xj**3.0)
        names.append(f'x{j}^3')
    
    # Interaction: (x_i + x_j)
    for i in range(d):
        for j in range(i+1, d):
            cols.append(X[:, i] + X[:, j])
            names.append(f'x{i}+x{j}')
            cols.append(X[:, i] - X[:, j])
            names.append(f'x{i}-x{j}')
            cols.append(X[:, i] + X[:, j])**2
            names.append(f'(x{i}+x{j})^2')
            cols.append((X[:, i] - X[:, j])**2)
            names.append(f'(x{i}-x{j})^2')
    
    # Normalize columns to improve numerical stability
    lib = np.column_stack(cols)
    # Replace inf/nan
    lib = np.nan_to_num(lib, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Column normalization
    col_norms = np.linalg.norm(lib, axis=0)
    col_norms[col_norms == 0] = 1.0
    lib = lib / col_norms
    
    return lib, names


def _sparse_regression(lib, y, max_iter=100, tol=1e-8):
    """
    Perform sparse regression using iterative thresholding (SINDy-like).
    Returns coefficients.
    """
    n, m = lib.shape
    theta = np.zeros(m)
    
    # Start with dense least squares
    theta, *_ = lstsq(lib, y, cond=None)
    
    # Iterative thresholding
    for it in range(max_iter):
        # Compute residual
        residual = y - lib @ theta
        
        # For each column, compute correlation with residual
        # Use gradient step
        # theta_new = theta + alpha * lib.T @ residual / n
        alpha = 1.0 / np.max(np.linalg.norm(lib, axis=0)**2 + 1e-10)
        grad = lib.T @ residual / n
        theta_new = theta + alpha * grad
        
        # Threshold: keep only top k coefficients
        # Adaptive k: start with all, prune
        abs_theta = np.abs(theta_new)
        # Keep top 20% or at least 5 terms
        k = max(5, m // 5)
        if m <= 10:
            k = m
        
        # Threshold small coefficients
        threshold = np.percentile(abs_theta, 100 - 100.0 * k / m)
        mask = abs_theta > threshold
        if mask.sum() < 5:
            # Keep at least 5 largest
            idx = np.argsort(abs_theta)[-5:]
            mask = np.zeros(m, dtype=bool)
            mask[idx] = True
        
        theta_new[~mask] = 0.0
        
        # Check convergence
        if np.linalg.norm(theta_new - theta) < tol * (np.linalg.norm(theta) + 1e-10):
            theta = theta_new
            break
        theta = theta_new
    
    return theta


def _evaluate_mse(predict, X, y):
    """Compute normalized MSE."""
    pred = predict(X)
    mse = np.mean((pred - y)**2)
    var_y = np.var(y)
    if var_y < 1e-15:
        var_y = 1.0
    return mse / var_y


def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290  # Leave 40s margin
    
    n, d = X_train.shape
    y_train = np.asarray(y_train, dtype=float).ravel()
    
    # Build library
    lib, names = _build_library(X_train)
    m = lib.shape[1]
    
    best_theta = None
    best_mse = np.inf
    
    # Try multiple sparsity levels via iterative thresholding with different k
    for k_frac in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]:
        if time.time() > deadline:
            break
        
        k = max(1, int(m * k_frac))
        
        # Use LARS-like approach: start from dense, iteratively zero out
        # For now, use the sparse regression function with modified k
        theta = np.zeros(m)
        
        # Dense solution first
        theta_dense, *_ = lstsq(lib, y_train, cond=None)
        
        # Iteratively prune
        theta_current = theta_dense.copy()
        for prune_iter in range(30):
            if time.time() > deadline:
                break
            
            residual = y_train - lib @ theta_current
            abs_theta = np.abs(theta_current)
            
            # Determine threshold
            if k < m:
                threshold = np.percentile(abs_theta, 100 - 100.0 * k / m)
                mask = abs_theta > threshold
                if mask.sum() < max(1, k // 2):
                    idx = np.argsort(abs_theta)[-max(1, k // 2):]
                    mask = np.zeros(m, dtype=bool)
                    mask[idx] = True
            else:
                mask = np.ones(m, dtype=bool)
            
            theta_pruned = theta_current.copy()
            theta_pruned[~mask] = 0.0
            
            # Refit on remaining terms
            if mask.sum() > 0:
                lib_sub = lib[:, mask]
                if lib_sub.shape[1] < n:
                    theta_sub, *_ = lstsq(lib_sub, y_train, cond=None)
                else:
                    theta_sub = np.linalg.pinv(lib_sub) @ y_train
                theta_new = np.zeros(m)
                theta_new[mask] = theta_sub
            else:
                theta_new = np.zeros(m)
            
            # Evaluate
            pred = lib @ theta_new
            mse = np.mean((pred - y_train)**2)
            var_y = np.var(y_train)
            if var_y < 1e-15:
                var_y = 1.0
            norm_mse = mse / var_y
            
            if norm_mse < best_mse:
                best_mse = norm_mse
                best_theta = theta_new.copy()
            
            if np.linalg.norm(theta_new - theta_current) < 1e-10:
                break
            theta_current = theta_new
    
    # If we have a good solution, use it
    if best_theta is None:
        # Fallback: dense least squares
        best_theta, *_ = lstsq(lib, y_train, cond=None)
    
    # Final prediction function
    lib_cached = lib
    theta_cached = best_theta.copy()
    
    def predict(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # Rebuild library for X
        lib_x, _ = _build_library(X)
        # Ensure same number of columns
        if lib_x.shape[1] != lib_cached.shape[1]:
            # This shouldn't happen, but handle it
            # Pad or truncate
            if lib_x.shape[1] < lib_cached.shape[1]:
                pad = np.zeros((lib_x.shape[0], lib_cached.shape[1] - lib_x.shape[1]))
                lib_x = np.hstack([lib_x, pad])
            else:
                lib_x = lib_x[:, :lib_cached.shape[1]]
        pred = lib_x @ theta_cached
        return pred
    
    return predict
