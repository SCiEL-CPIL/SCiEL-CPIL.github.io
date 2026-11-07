"""Symbolic regression over hidden equations. OMP with rich basis."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 300.0  # safety margin before 330s kill


def _time_left():
    return _DEADLINE - (time.time() - _T0)


def _build_basis(X):
    """Build a rich feature matrix for symbolic regression."""
    n, d = X.shape
    cols = [np.ones(n, dtype=float)]
    
    # Per-variable features
    for j in range(d):
        x = X[:, j].astype(float)
        cols.append(x)
        cols.append(x * x)
        cols.append(x * x * x)
        cols.append(x ** 4)
        cols.append(np.sin(x))
        cols.append(np.cos(x))
        cols.append(np.tanh(x))
        cols.append(np.log(np.abs(x) + 1e-9))
        cols.append(np.exp(-0.3 * x))
        cols.append(np.exp(-0.1 * x))
        cols.append(1.0 / (1.0 + np.exp(-x)))  # sigmoid
        cols.append(np.sqrt(np.abs(x) + 1e-9))
    
    # Pairwise products of raw inputs
    for i in range(d):
        for j in range(i + 1, d):
            cols.append(X[:, i] * X[:, j])
            cols.append(X[:, i] * X[:, i] * X[:, j])
            cols.append(X[:, i] * X[:, j] * X[:, j])
            cols.append(np.sin(X[:, i]) * np.cos(X[:, j]))
            cols.append(np.cos(X[:, i]) * np.sin(X[:, j]))
            cols.append(np.tanh(X[:, i]) * np.tanh(X[:, j]))
    
    # Triple products for d >= 3
    if d >= 3:
        for i in range(d):
            for j in range(i + 1, d):
                for k in range(j + 1, d):
                    cols.append(X[:, i] * X[:, j] * X[:, k])
    
    B = np.column_stack(cols)
    # normalize columns for numerical stability
    scale = np.where(np.abs(B).max(axis=0) > 0, np.abs(B).max(axis=0), 1.0)
    Bn = B / scale
    return Bn, scale


def _omp_select(Bn, y, max_features=30, reg=1e-8):
    """Orthogonal Matching Pursuit for feature selection."""
    n, p = Bn.shape
    ymean = float(np.mean(y))
    yc = y - ymean
    
    # Center Bn columns
    Bn_centered = Bn.copy()
    for j in range(p):
        Bn_centered[:, j] -= np.mean(Bn_centered[:, j])
    
    # Precompute norms
    col_norms = np.sqrt(np.sum(Bn_centered * Bn_centered, axis=0))
    col_norms[col_norms < 1e-12] = 1e-12
    
    # Greedy selection
    selected = []
    residual = yc.copy()
    residual_norm_sq = float(np.sum(residual * residual))
    
    for _ in range(max_features):
        if _time_left() < 2.0:
            break
        if len(selected) >= p:
            break
            
        # Find best feature by correlation with residual
        corr = Bn_centered.T @ residual
        scores = np.abs(corr) / (col_norms * np.sqrt(residual_norm_sq + 1e-12))
        
        # Mask already selected
        for j in selected:
            scores[j] = -1
        
        best_j = int(np.argmax(scores))
        if scores[best_j] < 1e-6:
            break
        
        selected.append(best_j)
        
        # Refit on selected features
        if len(selected) == 1:
            # Single feature: w = corr / norm^2
            w = corr[best_j] / (col_norms[best_j] ** 2)
            residual = yc - w * Bn_centered[:, best_j]
        else:
            # Refit using lstsq on selected columns
            Bs = Bn_centered[:, selected]
            try:
                w_vec, _, _, _ = np.linalg.lstsq(Bs, yc, rcond=None)
            except:
                break
            residual = yc - Bs @ w_vec
        
        residual_norm_sq = float(np.sum(residual * residual))
        if residual_norm_sq < 1e-10 * float(np.sum(yc * yc)):
            break
    
    return selected, ymean


def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    
    Bn, scale = _build_basis(X_train)
    p = Bn.shape[1]
    
    # Use OMP for feature selection
    selected, ymean = _omp_select(Bn, y_train, max_features=min(40, p), reg=1e-8)
    
    if not selected:
        selected = [0]
    
    # Refit with regularization on selected features
    Bs = Bn[:, selected]
    y = y_train
    
    # Add small ridge regularization
    reg = 1e-6
    G = Bs.T @ Bs
    G.flat[::Bs.shape[1] + 1] += reg
    rhs = Bs.T @ y
    try:
        w = np.linalg.solve(G, rhs)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(Bs, y, rcond=None)[0]
    
    def predict(X):
        X = np.asarray(X, dtype=float)
        Bn_x, _ = _build_basis(X)
        pred = Bn_x[:, selected] @ w
        return pred
    
    return predict
# EVOLVE-BLOCK-END
