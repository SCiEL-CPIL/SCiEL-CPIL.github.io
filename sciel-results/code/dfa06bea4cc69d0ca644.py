"""Sparse Kernel Regression with Nyström approximation and elastic-net selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Global time tracking
_TIME_START = [0.0]
_TOTAL_DEADLINE = 320.0  # Leave 10s margin before 330s kill

def _remaining():
    return _TOTAL_DEADLINE - (time.time() - _TIME_START[0])

def _poly_basis(X, degree):
    """Return a basis matrix of all monomials up to `degree` (incl. constant)."""
    n, d = X.shape
    cols = [np.ones(n)]
    for deg in range(1, degree + 1):
        for r in range(d):
            for c in range(r, d):
                term = np.ones(n)
                for k in range(r, c + 1):
                    term = term * X[:, k]
                cols.append(term)
    return np.column_stack(cols)

def _rbf_kernel(X, Y, gamma):
    """Compute RBF kernel between X and Y."""
    d2 = (np.sum(X ** 2, axis=1)[:, None] + np.sum(Y ** 2, axis=1)[None, :] - 2 * X @ Y.T)
    d2 = np.maximum(d2, 0.0)
    return np.exp(-gamma * d2)

def _nystrom_landmarks(X, n_landmarks, gamma, seed=42):
    """Select diverse landmarks using greedy farthest-point sampling on RBF kernel."""
    n = X.shape[0]
    if n_landmarks >= n:
        return np.arange(n)
    
    # Compute RBF kernel
    K = _rbf_kernel(X, X, gamma)
    
    # Greedy selection: start with a random point, then pick points that maximize
    # the minimum distance to already selected landmarks
    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(n))]
    
    for _ in range(n_landmarks - 1):
        # Compute kernel similarity to each selected landmark
        # We want to maximize the minimum kernel value (i.e., minimize distance)
        # Actually, for diversity we want to maximize the minimum distance
        # Use the kernel to compute "distance" in feature space
        # Simpler: use Euclidean distance
        X_sel = X[selected]
        # Compute squared distances from all points to selected landmarks
        d2 = (np.sum(X ** 2, axis=1)[:, None] + np.sum(X_sel ** 2, axis=1)[None, :] - 2 * X @ X_sel.T)
        d2 = np.maximum(d2, 0.0)
        min_dist = np.min(d2, axis=1)
        
        # Exclude already selected points
        for idx in selected:
            min_dist[idx] = -1
        
        # Pick the point with maximum minimum distance
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
    
    return np.array(selected)

def _fit_sparse_kernel(X, y, gamma, n_landmarks, alpha_l1, alpha_l2, seed=42):
    """
    Fit sparse kernel regression using Nyström approximation.
    
    1. Select n_landmarks diverse landmarks from training data
    2. Compute Nystrom approximation of the kernel
    3. Fit coefficients via elastic-net regularization
    """
    n, d = X.shape
    n_landmarks = min(n_landmarks, n)
    
    # Select landmarks
    landmarks_idx = _nystrom_landmarks(X, n_landmarks, gamma, seed)
    X_land = X[landmarks_idx]
    
    # Compute Nystrom approximation
    # K_nystrom(X, X') = K(X, X_land) @ K(X_land, X_land)^{-1} @ K(X_land, X')
    K_land_land = _rbf_kernel(X_land, X_land, gamma)
    # Add small regularization for numerical stability
    K_land_land += 1e-8 * np.eye(n_landmarks)
    
    try:
        K_land_land_inv = np.linalg.inv(K_land_land)
    except np.linalg.LinAlgError:
        K_land_land_inv = np.linalg.pinv(K_land_land)
    
    # Precompute K_land_land_inv @ y_land
    y_land = y[landmarks_idx]
    alpha_land = K_land_land_inv @ y_land
    
    def predict(Xt):
        K_xt_land = _rbf_kernel(Xt, X_land, gamma)
        # Nystrom prediction: K(Xt, X_land) @ K(X_land, X_land)^{-1} @ y_land
        return K_xt_land @ alpha_land
    
    # Now fit with elastic-net: we want to find coefficients w such that
    # y ≈ K_nystrom(X, X) @ w, where K_nystrom(X, X) = K(X, X_land) @ K_land_land_inv @ K(X_land, X)
    # Let Z = K(X, X_land) @ K_land_land_inv @ K(X_land, X)  (n x n)
    # But this is expensive. Instead, use the dual form:
    # y ≈ K(X, X_land) @ alpha, where alpha is n_landmarks
    # Fit alpha via elastic-net on the projected features
    K_x_land = _rbf_kernel(X, X_land, gamma)  # n x n_landmarks
    
    # Elastic-net: minimize ||y - K_x_land @ alpha||^2 + alpha_l1 * ||alpha||_1 + alpha_l2 * ||alpha||^2
    # Use scipy's minimize with L1 penalty (soft-thresholding via projected gradient)
    # Simpler: use L2 only (ridge) for stability, then apply soft-thresholding
    # For now, use ridge regression on the Nystrom features
    A = K_x_land.T @ K_x_land + (alpha_l2 + 1e-8) * np.eye(n_landmarks)
    b = K_x_land.T @ y
    alpha = np.linalg.solve(A, b)
    
    # Apply soft-thresholding for L1 sparsity
    if alpha_l1 > 0:
        alpha = np.sign(alpha) * np.maximum(np.abs(alpha) - alpha_l1, 0)
    
    def predict_final(Xt):
        K_xt_land = _rbf_kernel(Xt, X_land, gamma)
        return K_xt_land @ alpha
    
    return predict_final

def _fit_ridge_rbf(X, y, gamma, lam):
    """Standard RBF kernel ridge regression."""
    K = _rbf_kernel(X, X, gamma)
    n = K.shape[0]
    alpha = np.linalg.solve(K + lam * np.eye(n), y)
    def predict(Xt):
        return _rbf_kernel(Xt, X, gamma) @ alpha
    return predict

def _fit_poly_ridge(X, y, degree, lam):
    """Polynomial ridge regression."""
    B = _poly_basis(X, degree)
    n, m = B.shape
    A = B.T @ B + lam * np.eye(m)
    w = np.linalg.solve(A, B.T @ y)
    return lambda Xt: _poly_basis(Xt, degree) @ w

def _fit_gp_m52(X, y, max_iter=50):
    """Matern-5/2 GP regression with hyperparameter optimization."""
    n, d = X.shape
    
    def build_K(theta):
        l, sigma_f, sigma_n = theta
        l = max(l, 1e-4)
        d2 = (np.sum(X ** 2, axis=1)[:, None] + np.sum(X ** 2, axis=1)[None, :] - 2 * X @ X.T)
        d2 = np.maximum(d2, 0.0)
        r = np.sqrt(d2 + 1e-12)
        K = (sigma_f ** 2) * (1 + np.sqrt(5) * r + 5 / 3 * r ** 2) * np.exp(-np.sqrt(5) * r / l)
        K[np.diag_indices(n)] += sigma_n ** 2
        return K
    
    def log_marginal(theta):
        K = build_K(theta)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        lmc = 0.5 * np.sum(y * alpha) + np.sum(np.log(np.diag(L))) + 0.5 * n * np.log(2 * np.pi)
        return lmc
    
    scale = np.max(np.ptp(X, axis=0)) + 1e-6
    yvar = np.var(y) + 1e-9
    theta0 = np.array([scale, np.sqrt(yvar), np.sqrt(yvar) * 1e-2])
    
    res = minimize(log_marginal, theta0, method='Nelder-Mead',
                   options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4})
    l, sigma_f, sigma_n = res.x
    l = max(l, 1e-4)
    K = build_K(res.x)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    
    def predict(Xt):
        d2 = (np.sum(Xt ** 2, axis=1)[:, None] + np.sum(X ** 2, axis=1)[None, :] - 2 * Xt @ X.T)
        d2 = np.maximum(d2, 0.0)
        r = np.sqrt(d2 + 1e-12)
        Ks = (sigma_f ** 2) * (1 + np.sqrt(5) * r + 5 / 3 * r ** 2) * np.exp(-np.sqrt(5) * r / l)
        return Ks @ alpha
    return predict

def fit(X_train, y_train):
    _TIME_START[0] = time.time()
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    n, d = X.shape
    yvar = np.var(y) + 1e-12
    
    candidates = []
    
    # 1) Polynomial baselines
    for deg in (2, 3, 4):
        if _remaining() < 30:
            break
        for lam in (1e-6, 1e-4, 1e-2):
            try:
                p = _fit_poly_ridge(X, y, deg, lam)
                candidates.append((f'poly{deg}_l{lam}', p))
            except Exception:
                pass
    
    # 2) Standard RBF KRR
    if _remaining() > 60:
        for gamma in (0.1, 0.5, 1.0, 2.0, 5.0):
            for lam in (1e-3, 1e-2, 1e-1):
                try:
                    p = _fit_ridge_rbf(X, y, gamma, lam)
                    candidates.append((f'rbf_g{gamma}_l{lam}', p))
                except Exception:
                    pass
    
    # 3) Sparse Kernel Regression (Nyström) - the structural innovation
    if _remaining() > 90:
        n_landmarks_options = [10, 20, 50]
        for n_land in n_landmarks_options:
            for gamma in (0.5, 1.0, 2.0):
                for alpha_l1 in (0.0, 0.01, 0.1):
                    for alpha_l2 in (1e-3, 1e-2):
                        if _remaining() < 30:
                            break
                        try:
                            p = _fit_sparse_kernel(X, y, gamma, n_land, alpha_l1, alpha_l2)
                            candidates.append((f'sparse_g{gamma}_nl{n_land}_l1{alpha_l1}_l2{alpha_l2}', p))
                        except Exception:
                            continue
    
    # 4) GP Matern-5/2
    if _remaining() > 60:
        try:
            p = _fit_gp_m52(X, y, max_iter=40)
            candidates.append(('gp_m52', p))
        except Exception:
            pass
    
    if not candidates:
        c = float(np.mean(y))
        return lambda Xt: np.full(len(Xt), c)
    
    # Select best by training residual
    best_name, best_pred, best_score = None, None, np.inf
    for name, p in candidates:
        try:
            pred = p(X)
            score = np.mean((pred - y) ** 2) / yvar
            if score < best_score:
                best_score, best_name, best_pred = score, name, p
        except Exception:
            continue
    
    if best_pred is None:
        c = float(np.mean(y))
        return lambda Xt: np.full(len(Xt), c)
    
    final = best_pred
    
    def predict(Xt):
        Xt = np.asarray(Xt, dtype=float)
        out = final(Xt)
        out = np.asarray(out, dtype=float).ravel()
        out = np.where(np.isfinite(out), out, np.nanmean(y))
        return out
    
    return predict
# EVOLVE-BLOCK-END
