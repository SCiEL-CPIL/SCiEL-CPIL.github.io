"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _fit_poly(x, y, deg):
    """Fit a polynomial of given degree, return coefficients padded to length 6."""
    try:
        c = np.polynomial.polynomial.polyfit(x, y, deg)
        if not np.all(np.isfinite(c)):
            raise ValueError("Non-finite")
        out = np.zeros(6)
        out[:deg + 1] = c
        return out
    except Exception:
        return None


def _ridge_cv(x, y, deg, lam, n_folds=5, seed=42):
    """
    K-fold cross-validation with ridge regularization on higher-order terms.
    Returns mean validation MSE.
    """
    n = len(x)
    if n < n_folds * 2:
        n_folds = max(2, n // 2)
    if n_folds < 2:
        # Too little data; use training error
        V = np.polynomial.polynomial.polyvander(x, deg)
        reg_w = np.array([1.0] + [lam * (i + 1) ** 2 for i in range(1, deg + 1)])
        R = np.diag(reg_w)
        try:
            A = V.T @ V + R
            b = V.T @ y
            c = np.linalg.solve(A, b)
            if not np.all(np.isfinite(c)):
                return np.inf
            pred = np.polynomial.polynomial.polyval(x, c)
            return np.mean((pred - y) ** 2)
        except Exception:
            return np.inf

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    fold_size = n // n_folds
    total_mse = 0.0
    valid_folds = 0

    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else n
        val_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])

        if len(train_idx) < deg + 1:
            continue

        x_tr = x[train_idx]
        y_tr = y[train_idx]
        V = np.polynomial.polynomial.polyvander(x_tr, deg)

        reg_w = np.array([1.0] + [lam * (i + 1) ** 2 for i in range(1, deg + 1)])
        R = np.diag(reg_w)

        try:
            A = V.T @ V + R
            b = V.T @ y_tr
            c = np.linalg.solve(A, b)
            if not np.all(np.isfinite(c)):
                continue
        except Exception:
            continue

        pred = np.polynomial.polynomial.polyval(x[val_idx], c)
        total_mse += np.mean((pred - y[val_idx]) ** 2)
        valid_folds += 1

    if valid_folds == 0:
        return np.inf
    return total_mse / valid_folds


def _fista_lasso(A, b, lam, max_iter=2000, tol=1e-8):
    """
    FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) for Lasso.
    Minimize 0.5*||Ax-b||^2 + lam*||c||_1
    """
    n = len(b)
    p = A.shape[1]
    
    # Precompute A^T A and A^T b for efficiency
    AtA = A.T @ A
    Atb = A.T @ b
    
    # Lipschitz constant
    L = np.linalg.norm(AtA, ord=2)
    if L < 1e-12:
        L = 1.0
    step = 1.0 / L
    
    # Initialize
    c = np.zeros(p)
    c_prev = np.zeros(p)
    t = 1.0
    
    for _ in range(max_iter):
        t_prev = t
        c_new = c - step * (AtA @ c - Atb)
        c_new = np.sign(c_new) * np.maximum(np.abs(c_new) - step * lam, 0.0)
        
        # Convergence check
        if np.linalg.norm(c_new - c) < tol * (1 + np.linalg.norm(c)):
            c = c_new
            break
        
        t = 0.5 * (1 + np.sqrt(1 + 4 * t_prev ** 2))
        c = c_new + ((t_prev - 1) / t) * (c_new - c_prev)
        c_prev = c_new
    
    return c


def _lasso_cv(x, y, deg, lam, n_folds=5, seed=42):
    """
    K-fold cross-validation with L1 (Lasso) regularization.
    Returns mean validation MSE.
    """
    n = len(x)
    if n < n_folds * 2:
        n_folds = max(2, n // 2)
    if n_folds < 2:
        V = np.polynomial.polynomial.polyvander(x, deg)
        c = _fista_lasso(V, y, lam, max_iter=1000)
        pred = np.polynomial.polynomial.polyval(x, c)
        return np.mean((pred - y) ** 2)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    fold_size = n // n_folds
    total_mse = 0.0
    valid_folds = 0

    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else n
        val_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])

        if len(train_idx) < deg + 1:
            continue

        x_tr = x[train_idx]
        y_tr = y[train_idx]
        V = np.polynomial.polynomial.polyvander(x_tr, deg)

        c = _fista_lasso(V, y_tr, lam, max_iter=1500)
        if not np.all(np.isfinite(c)):
            continue

        pred = np.polynomial.polynomial.polyval(x[val_idx], c)
        total_mse += np.mean((pred - y[val_idx]) ** 2)
        valid_folds += 1

    if valid_folds == 0:
        return np.inf
    return total_mse / valid_folds


def construct(training_x, training_subgrid_tendency):
    t0 = time.time()
    deadline = t0 + 110.0

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y)
    if not np.all(mask):
        x = x[mask]
        y = y[mask]

    if len(x) < 6:
        return np.zeros(6)

    best_coeff = None
    best_score = np.inf

    # First: try standard polynomial fits with ridge regularization
    reg_lambdas = [0.0, 1e-4, 1e-3, 1e-2, 0.1]
    
    for deg in range(6):
        if time.time() > deadline - 10.0:
            break

        for reg_lam in reg_lambdas:
            if time.time() > deadline - 5.0:
                break

            if reg_lam == 0.0:
                score = _ridge_cv(x, y, deg, 0.0)
            else:
                score = _ridge_cv(x, y, deg, reg_lam)

            if score < best_score:
                best_score = score
                if reg_lam == 0.0:
                    best_coeff = _fit_poly(x, y, deg)
                else:
                    V = np.polynomial.polynomial.polyvander(x, deg)
                    reg_w = np.array([1.0] + [reg_lam * (i + 1) ** 2 for i in range(1, deg + 1)])
                    R = np.diag(reg_w)
                    try:
                        A = V.T @ V + R
                        b = V.T @ y
                        c = np.linalg.solve(A, b)
                        if np.all(np.isfinite(c)):
                            best_coeff = np.zeros(6)
                            best_coeff[:deg + 1] = c
                    except Exception:
                        best_coeff = _fit_poly(x, y, deg)

    # Second: try Lasso for sparse coefficient selection
    if time.time() < deadline - 15.0:
        lasso_lams = [1e-4, 1e-3, 1e-2, 0.1, 1.0]
        for deg in range(3, 6):  # Focus on higher degrees where sparsity matters
            if time.time() > deadline - 5.0:
                break

            for lam in lasso_lams:
                if time.time() > deadline - 3.0:
                    break

                score = _lasso_cv(x, y, deg, lam)

                if score < best_score:
                    best_score = score
                    V = np.polynomial.polynomial.polyvander(x, deg)
                    c = _fista_lasso(V, y, lam, max_iter=3000)
                    if np.all(np.isfinite(c)):
                        best_coeff = np.zeros(6)
                        best_coeff[:deg + 1] = c

    if best_coeff is not None and np.all(np.isfinite(best_coeff)):
        return best_coeff

    # Fallback: linear
    coeff = _fit_poly(x, y, 1)
    if coeff is not None:
        return coeff

    return np.zeros(6)
# EVOLVE-BLOCK-END
