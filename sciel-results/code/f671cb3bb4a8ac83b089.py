"""Symbolic regression over hidden equations using OMP feature selection + rich basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _time_left(deadline):
    return max(0.0, deadline - time.time())

def _build_basis(X):
    """Build a rich basis matrix from input X (n, d)."""
    n, d = X.shape
    cols = [np.ones(n)]
    names = ['1']
    
    # Linear terms
    for j in range(d):
        cols.append(X[:, j])
        names.append(f'x{j}')
    
    # Quadratic and cubic powers
    for j in range(d):
        xj = X[:, j]
        cols.append(xj ** 2)
        names.append(f'x{j}^2')
        cols.append(xj ** 3)
        names.append(f'x{j}^3')
        cols.append(xj ** 4)
        names.append(f'x{j}^4')
        cols.append(xj ** 5)
        names.append(f'x{j}^5')
    
    # Cross terms (quadratic)
    for i in range(d):
        for j in range(i + 1, d):
            cols.append(X[:, i] * X[:, j])
            names.append(f'x{i}*x{j}')
    
    # Cross terms (cubic)
    for i in range(d):
        for j in range(i + 1, d):
            for k in range(j + 1, d):
                cols.append(X[:, i] * X[:, j] * X[:, k])
                names.append(f'x{i}*x{j}*x{k}')
    
    # Trig functions
    for j in range(d):
        xj = X[:, j]
        cols.append(np.sin(xj))
        names.append(f'sin(x{j})')
        cols.append(np.cos(xj))
        names.append(f'cos(x{j})')
        cols.append(np.sin(2 * xj))
        names.append(f'sin(2*x{j})')
        cols.append(np.cos(2 * xj))
        names.append(f'cos(2*x{j})')
    
    # Exp and log
    for j in range(d):
        xj = X[:, j]
        cols.append(np.exp(xj))
        names.append(f'exp(x{j})')
        cols.append(np.log(xj))
        names.append(f'log(x{j})')
        cols.append(np.sqrt(xj))
        names.append(f'sqrt(x{j})')
        cols.append(np.abs(xj))
        names.append(f'abs(x{j})')
        cols.append(1.0 / (xj + 1e-8))
        names.append(f'1/(x{j}+eps)')
    
    # Pairwise trig products
    for i in range(d):
        for j in range(i + 1, d):
            cols.append(np.sin(X[:, i]) * np.cos(X[:, j]))
            names.append(f'sin(x{i})*cos(x{j})')
            cols.append(np.cos(X[:, i]) * np.sin(X[:, j]))
            names.append(f'cos(x{i})*sin(x{j})')
    
    # Pairwise exp/log combos
    for i in range(d):
        for j in range(i + 1, d):
            cols.append(np.exp(X[:, i]) * np.exp(X[:, j]))
            names.append(f'exp(x{i})*exp(x{j})')
            cols.append(np.log(X[:, i]) * np.log(X[:, j]))
            names.append(f'log(x{i})*log(x{j})')
    
    # Min/max differences
    for i in range(d):
        for j in range(i + 1, d):
            diff = X[:, i] - X[:, j]
            cols.append(diff)
            names.append(f'x{i}-x{j}')
            cols.append(np.abs(diff))
            names.append(f'|x{i}-x{j}|')
            cols.append(np.maximum(X[:, i], X[:, j]))
            names.append(f'max(x{i},x{j})')
            cols.append(np.minimum(X[:, i], X[:, j]))
            names.append(f'min(x{i},x{j})')
    
    basis = np.column_stack(cols)
    # Remove columns with NaN/Inf
    mask = np.all(np.isfinite(basis), axis=0)
    basis = basis[:, mask]
    names = [n for n, m in zip(names, mask) if m]
    
    return basis, names

def _omp_select(basis, y, max_features, tol=1e-10, deadline=None):
    """Orthogonal Matching Pursuit to select features."""
    n, p = basis.shape
    y_resid = y.copy()
    y_norm = np.linalg.norm(y_resid)
    if y_norm == 0:
        return [0]
    
    selected = []
    remaining = list(range(p))
    
    # Pre-compute norms of basis vectors
    basis_norms = np.linalg.norm(basis, axis=0)
    basis_norms[basis_norms == 0] = 1e-10
    
    # Normalize basis for OMP
    basis_norm = basis / basis_norms
    
    for step in range(max_features):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute correlations
        corr = np.abs(basis_norm[:, remaining] @ y_resid) / (np.linalg.norm(y_resid) + 1e-10)
        
        if len(corr) == 0:
            break
        
        best_idx = np.argmax(corr)
        if corr[best_idx] < tol:
            break
        
        selected.append(remaining[best_idx])
        remaining.pop(best_idx)
        
        # Update residual
        S = np.array(selected)
        if len(S) > 0:
            try:
                coeffs, *_ = np.linalg.lstsq(basis[:, S], y, rcond=None)
                y_resid = y - basis[:, S] @ coeffs
            except:
                break
        
        if np.linalg.norm(y_resid) < tol * y_norm:
            break
    
    return selected

def _fit_ridge(basis, y, alpha=1e-6):
    """Ridge regression."""
    n, p = basis.shape
    A = basis.T @ basis
    b = basis.T @ y
    A += alpha * np.eye(p)
    try:
        w = np.linalg.solve(A, b)
    except:
        w, *_ = np.linalg.lstsq(A, b, rcond=None)
    return w

def fit(X_train, y_train):
    deadline = time.time() + 280  # Leave 20s margin before 300s limit
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    
    d = X.shape[1]
    n = X.shape[0]
    
    # Build basis
    basis, names = _build_basis(X)
    p = basis.shape[1]
    
    # Standardize basis for better numerical stability
    basis_mean = basis.mean(axis=0)
    basis_std = basis.std(axis=0)
    basis_std[basis_std < 1e-10] = 1.0
    basis_std = basis_std  # Keep original scale for prediction, use standardized for fitting
    
    # Use OMP to select features
    max_feat = min(p, 50)
    selected = _omp_select(basis, y, max_feat, tol=1e-8, deadline=deadline)
    
    if len(selected) == 0:
        selected = [0]
    
    # Fit ridge regression on selected features
    B = basis[:, selected]
    alpha = 1e-6
    w = _fit_ridge(B, y, alpha)
    
    def predict(X_test):
        X_t = np.asarray(X_test, dtype=np.float64)
        if X_t.ndim == 1:
            X_t = X_t.reshape(1, -1)
        B_test, _ = _build_basis(X_t)
        pred = B_test[:, selected] @ w
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        return pred
    
    return predict
# EVOLVE-BLOCK-END
