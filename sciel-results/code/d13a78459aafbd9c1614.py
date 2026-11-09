"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar


def _build_features(X):
    """Build a rich feature set covering polynomials, trig, exp, log, reciprocal, and interactions."""
    n = X.shape[0]
    d = X.shape[1]
    cols = [np.ones(n)]
    
    for j in range(d):
        xj = X[:, j]
        # Powers up to degree 6
        for p in range(1, 7):
            cols.append(xj ** p)
        # Trig features
        cols.append(np.sin(xj))
        cols.append(np.cos(xj))
        cols.append(np.sin(2 * xj))
        cols.append(np.cos(2 * xj))
        # Exponential
        cols.append(np.exp(-xj))
        cols.append(np.exp(-0.5 * xj))
        # Log
        cols.append(np.log(xj + 1e-8))
        # Reciprocal
        cols.append(1.0 / (xj + 1e-8))
        cols.append(1.0 / (xj ** 2 + 1e-8))
        # x * trig
        cols.append(xj * np.sin(xj))
        cols.append(xj * np.cos(xj))
        # sqrt
        cols.append(np.sqrt(np.abs(xj)))
    
    # Pairwise products (degree 2 cross terms)
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
    
    # Triple products for small d
    if d <= 3:
        for i in range(d):
            for j in range(i, d):
                for k in range(j, d):
                    cols.append(X[:, i] * X[:, j] * X[:, k])
    
    # Sum and product of all inputs
    if d >= 2:
        cols.append(np.sum(X, axis=1))
        cols.append(np.prod(X, axis=1))
        # Pairwise sums
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(X[:, i] + X[:, j])
    
    F = np.column_stack(cols)
    # Replace any non-finite values
    F = np.nan_to_num(F, nan=0.0, posinf=1e6, neginf=-1e6)
    return F


def _solve_ridge(F, y, lam):
    """Solve ridge regression: (F^T F + lam*I) w = F^T y"""
    n_feat = F.shape[1]
    A = F.T @ F + lam * np.eye(n_feat)
    b = F.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w, *_ = np.linalg.lstsq(A, b, rcond=None)
    return w


def _solve_lasso(F, y, alpha, max_iter=5000, tol=1e-6):
    """Simple coordinate-descent LASSO."""
    n, p = F.shape
    w = np.zeros(p)
    F_norm_sq = np.sum(F ** 2, axis=0)
    F_norm_sq[F_norm_sq < 1e-12] = 1e-12
    
    for it in range(max_iter):
        max_change = 0.0
        for j in range(p):
            w_old = w[j]
            # Compute residual excluding feature j
            r = y - F @ w + F[:, j] * w[j]
            # Soft threshold
            rho = np.dot(F[:, j], r) / n
            w_new = np.sign(rho) * max(abs(rho) - alpha, 0.0) / F_norm_sq[j]
            w[j] = w_new
            change = abs(w_new - w_old)
            if change > max_change:
                max_change = change
        if max_change < tol:
            break
    
    return w


def fit(X_train, y_train):
    t0 = time.time()
    deadline = t0 + 280.0
    
    F_train = _build_features(X_train)
    n, p = F_train.shape
    
    # Standardize features (keep intercept column)
    F_std = F_train.copy()
    mu = np.zeros(p)
    sigma = np.ones(p)
    for j in range(1, p):
        mu[j] = np.mean(F_std[:, j])
        sigma[j] = np.std(F_std[:, j])
        if sigma[j] < 1e-12:
            sigma[j] = 1.0
        F_std[:, j] = (F_std[:, j] - mu[j]) / sigma[j]
    
    # Split into train/val for model selection
    rng = np.random.RandomState(42)
    idx = rng.permutation(n)
    n_val = max(20, n // 5)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    
    F_tr, y_tr = F_std[tr_idx], y_train[tr_idx]
    F_val, y_val = F_std[val_idx], y_train[val_idx]
    
    best_val_mse = np.inf
    best_w = None
    best_type = "ridge"
    best_param = 1.0
    
    # Try ridge with various lambdas (on standardized features)
    for lam in [1e-8, 1e-6, 1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0, 100.0]:
        if time.time() > deadline:
            break
        w = _solve_ridge(F_tr, y_tr, lam)
        # Unstandardize: w_orig[j] = w_std[j] / sigma[j] for j>0
        w_orig = np.zeros(p)
        w_orig[0] = w[0] - np.sum(w[1:] * mu[1:] / sigma[1:])
        w_orig[1:] = w[1:] / sigma[1:]
        
        pred_val = F_train[val_idx] @ w_orig
        val_mse = np.mean((pred_val - y_val) ** 2)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_w = w_orig
            best_type = "ridge"
            best_param = lam
    
    # Try LASSO with various alphas
    if time.time() < deadline:
        for alpha in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
            if time.time() > deadline:
                break
            w_std = _solve_lasso(F_tr, y_tr, alpha, max_iter=3000, tol=1e-5)
            # Unstandardize
            w_orig = np.zeros(p)
            w_orig[0] = w_std[0] - np.sum(w_std[1:] * mu[1:] / sigma[1:])
            w_orig[1:] = w_std[1:] / sigma[1:]
            
            pred_val = F_train[val_idx] @ w_orig
            val_mse = np.mean((pred_val - y_val) ** 2)
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_w = w_orig
                best_type = "lasso"
                best_param = alpha
    
    # If validation didn't help (all MSEs high), try OLS on full data as fallback
    if best_val_mse > np.var(y_train) * 0.5:
        if time.time() < deadline:
            w_ols, *_ = np.linalg.lstsq(F_train, y_train, rcond=None)
            pred_val = F_train[val_idx] @ w_ols
            val_mse_ols = np.mean((pred_val - y_val) ** 2)
            if val_mse_ols < best_val_mse:
                best_w = w_ols
                best_val_mse = val_mse_ols
                best_type = "ols"
                best_param = 0.0
    
    # Final refinement: retrain on all data with the best regularization type/param
    if best_type == "ridge" and time.time() < deadline:
        w_final_std = _solve_ridge(F_std, y_train, best_param)
        w_final = np.zeros(p)
        w_final[0] = w_final_std[0] - np.sum(w_final_std[1:] * mu[1:] / sigma[1:])
        w_final[1:] = w_final_std[1:] / sigma[1:]
    elif best_type == "lasso" and time.time() < deadline:
        w_final_std = _solve_lasso(F_std, y_train, best_param, max_iter=5000, tol=1e-6)
        w_final = np.zeros(p)
        w_final[0] = w_final_std[0] - np.sum(w_final_std[1:] * mu[1:] / sigma[1:])
        w_final[1:] = w_final_std[1:] / sigma[1:]
    else:
        w_final = best_w.copy()
    
    w_final = np.nan_to_num(w_final, nan=0.0, posinf=0.0, neginf=0.0)
    
    def predict(X):
        F = _build_features(X)
        pred = F @ w_final
        return np.nan_to_num(pred, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return predict
# EVOLVE-BLOCK-END
