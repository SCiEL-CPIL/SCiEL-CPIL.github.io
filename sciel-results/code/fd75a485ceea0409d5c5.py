"""
Combined approach: Full KRR with RBF + Polynomial kernels, cross-validated,
plus a symbolic regression attempt for low-dimensional problems.
"""
import numpy as np
import time
from itertools import product

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 295.0  # Safety margin

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    y_mean = np.mean(y)
    y_std = np.std(y)

    if y_std < 1e-10:
        def predict_const(X):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            return np.full(X.shape[0], y_mean)
        return predict_const

    # Normalize X for better kernel conditioning
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-10] = 1.0
    X_norm = (X - X_mean) / X_std

    def rbf_kernel(X1, X2, lengthscale):
        sq_dists = (np.sum(X1**2, axis=1)[:, None] + 
                    np.sum(X2**2, axis=1)[None, :] - 
                    2 * np.dot(X1, X2.T))
        np.maximum(sq_dists, 0, out=sq_dists)
        return np.exp(-sq_dists / (2 * lengthscale**2))

    def poly_kernel(X1, X2, degree, gamma=1.0):
        return (gamma * np.dot(X1, X2.T) + 1) ** degree

    def linear_kernel(X1, X2):
        return np.dot(X1, X2.T)

    # Cross-validation scoring
    def cv_score(kernel_func, alpha_reg):
        if time.time() > deadline - 5:
            return np.inf
        
        n = len(y)
        n_folds = 5
        fold_size = n // n_folds
        residuals = np.zeros(n)
        
        for fold in range(n_folds):
            if time.time() > deadline - 5:
                return np.inf
            
            val_idx = np.arange(fold * fold_size, min((fold + 1) * fold_size, n))
            trn_idx = np.setdiff1d(np.arange(n), val_idx)
            
            X_trn = X_norm[trn_idx]
            y_trn = y[trn_idx]
            X_val = X_norm[val_idx]
            y_val = y[val_idx]
            
            K = kernel_func(X_trn, X_trn)
            K[np.diag_indices_from(K)] += alpha_reg
            
            try:
                w = np.linalg.solve(K, y_trn)
                K_val = kernel_func(X_val, X_trn)
                pred_val = K_val @ w
                residuals[val_idx] = y_val - pred_val
            except np.linalg.LinAlgError:
                try:
                    w = np.linalg.pinv(K) @ y_trn
                    K_val = kernel_func(X_val, X_trn)
                    pred_val = K_val @ w
                    residuals[val_idx] = y_val - pred_val
                except Exception:
                    return np.inf
        
        mse = np.mean(residuals**2)
        return mse / (y_std**2)  # Normalized MSE

    # Generate candidate kernel configurations
    candidates = []
    
    # RBF kernels with various lengthscales
    for ls in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        candidates.append(('rbf', ls, lambda X1, X2, ls=ls: rbf_kernel(X1, X2, ls)))
    
    # Polynomial kernels
    for deg in [2, 3, 4]:
        for gamma in [0.5, 1.0, 2.0]:
            candidates.append((f'poly{deg}_g{gamma}', 0, 
                             lambda X1, X2, d=deg, g=gamma: poly_kernel(X1, X2, d, g)))
    
    # Linear
    candidates.append(('linear', 0, linear_kernel))
    
    # Combined kernels
    for ls in [0.3, 0.5, 1.0]:
        for deg in [2, 3]:
            candidates.append((f'combo_rbf{ls}_poly{deg}', 0,
                             lambda X1, X2, ls=ls, d=deg: 
                             rbf_kernel(X1, X2, ls) + poly_kernel(X1, X2, d, 1.0)))

    alphas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0]

    best_score = np.inf
    best_kernel_func = None
    best_alpha = None
    best_kernel_name = None

    # Evaluate candidates with cross-validation
    for name, _, kfunc in candidates:
        if time.time() > deadline - 30:
            break
        for alpha in alphas:
            if time.time() > deadline - 15:
                break
            score = cv_score(kfunc, alpha)
            if score < best_score:
                best_score = score
                best_kernel_func = kfunc
                best_alpha = alpha
                best_kernel_name = name

    # Final fit with best kernel
    if best_kernel_func is None:
        # Fallback: linear regression
        X_design = np.hstack([X, np.ones((n, 1))])
        try:
            w, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
            def predict_lin(X):
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                X_pred = np.hstack([X, np.ones((X.shape[0], 1))])
                return X_pred @ w
            return predict_lin
        except Exception:
            def predict_mean(X):
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                return np.full(X.shape[0], y_mean)
            return predict_mean

    K = best_kernel_func(X_norm, X_norm)
    K[np.diag_indices_from(K)] += best_alpha

    try:
        w = np.linalg.solve(K, y)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(K) @ y

    f_X_mean = X_mean
    f_X_std = X_std
    f_kernel_func = best_kernel_func
    f_w = w
    f_X_norm = X_norm

    def predict(X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_norm_test = (X - f_X_mean) / f_X_std
        K_test = f_kernel_func(X_norm_test, f_X_norm)
        return K_test @ f_w

    return predict
