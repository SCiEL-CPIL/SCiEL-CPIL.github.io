"""Symbolic regression via multi-kernel ridge with per-dimension length scales and efficient CV."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys


def _rbf_kernel(X1, X2, length_scales):
    """Compute RBF kernel between two sets of points with per-dimension length scales."""
    X1s = X1 / length_scales
    X2s = X2 / length_scales
    sq1 = np.sum(X1s**2, axis=1)[:, None]
    sq2 = np.sum(X2s**2, axis=1)[None, :]
    sq_dists = sq1 + sq2 - 2.0 * (X1s @ X2s.T)
    np.maximum(sq_dists, 0.0, out=sq_dists)
    return np.exp(-0.5 * sq_dists)


def _poly_kernel(X1, X2, degree, gamma, coef0):
    """Compute polynomial kernel."""
    return (gamma * (X1 @ X2.T) + coef0) ** degree


def _periodic_kernel(X1, X2, period, length_scale):
    """Compute periodic kernel."""
    diff = X1[:, None, :] - X2[None, :, :]
    sin_diff = np.sin(np.pi * diff / period)
    sq_sum = np.sum(sin_diff**2, axis=2)
    return np.exp(-2.0 * sq_sum / (length_scale**2))


def _compute_kernel(X1, X2, params):
    """
    Compute composite kernel: sum of RBF + polynomial + periodic kernels.
    params: dict with keys:
        - rbf_weight, rbf_ls (array of length d)
        - poly_weight, poly_degree, poly_gamma, poly_coef0
        - per_weight, per_period, per_ls
    """
    K = np.zeros((X1.shape[0], X2.shape[0]))
    
    if params.get('rbf_weight', 0) > 1e-12:
        K += params['rbf_weight'] * _rbf_kernel(X1, X2, params['rbf_ls'])
    
    if params.get('poly_weight', 0) > 1e-12:
        K += params['poly_weight'] * _poly_kernel(X1, X2, params['poly_degree'], 
                                                   params['poly_gamma'], params['poly_coef0'])
    
    if params.get('per_weight', 0) > 1e-12:
        K += params['per_weight'] * _periodic_kernel(X1, X2, params['per_period'], params['per_ls'])
    
    return K


def _kernel_ridge_solve(K, y, alpha_reg):
    """Solve kernel ridge regression: (K + alpha*I)^{-1} y"""
    n = K.shape[0]
    A = K + alpha_reg * np.eye(n)
    try:
        coeffs = np.linalg.solve(A, y)
    except np.linalg.LinAlgError:
        coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coeffs


def _cv_score(X_norm, y_centered, params, alpha, n_folds=5):
    """Compute cross-validation MSE for given parameters using leave-one-out style."""
    n = X_norm.shape[0]
    fold_size = n // n_folds
    indices = np.arange(n)
    
    # Precompute full kernel
    K_full = _compute_kernel(X_norm, X_norm, params)
    
    cv_mses = []
    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = min((fold + 1) * fold_size, n)
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
        
        K_train_fold = K_full[train_idx][:, train_idx]
        y_train_fold = y_centered[train_idx]
        y_val_fold = y_centered[val_idx]
        
        coeffs = _kernel_ridge_solve(K_train_fold, y_train_fold, alpha)
        
        K_val = K_full[val_idx][:, train_idx]
        pred_val = K_val @ coeffs
        mse = np.mean((pred_val - y_val_fold) ** 2)
        cv_mses.append(mse)
    
    return np.mean(cv_mses)


def fit(X_train, y_train):
    t0 = time.time()
    deadline = t0 + 310.0  # Leave 20s buffer for safety
    
    n, d = X_train.shape
    y_mean = np.mean(y_train)
    y_std = np.std(y_train) + 1e-12
    y_centered = y_train - y_mean
    
    # Normalize inputs to [0, 1]
    x_min = X_train.min(axis=0)
    x_max = X_train.max(axis=0)
    x_range = x_max - x_min + 1e-12
    X_norm = (X_train - x_min) / x_range
    
    best_params = None
    best_cv_mse = np.inf
    
    # Phase 1: Grid search over per-dimension RBF length scales and regularization
    # Start with isotropic, then try anisotropic for d > 1
    rbf_ls_grid = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    alpha_grid = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]
    
    # First pass: isotropic length scales
    for ls_val in rbf_ls_grid:
        if time.time() > deadline:
            break
        for alpha in alpha_grid:
            if time.time() > deadline:
                break
            
            params = {
                'rbf_weight': 1.0, 'rbf_ls': np.full(d, ls_val),
                'poly_weight': 0.0, 'poly_degree': 2, 'poly_gamma': 1.0, 'poly_coef0': 1.0,
                'per_weight': 0.0, 'per_period': 2.0, 'per_ls': 1.0
            }
            
            cv_mse = _cv_score(X_norm, y_centered, params, alpha)
            if cv_mse < best_cv_mse:
                best_cv_mse = cv_mse
                best_params = params.copy()
                best_params['alpha'] = alpha
    
    if best_params is None:
        best_params = {
            'rbf_weight': 1.0, 'rbf_ls': np.ones(d),
            'poly_weight': 0.0, 'poly_degree': 2, 'poly_gamma': 1.0, 'poly_coef0': 1.0,
            'per_weight': 0.0, 'per_period': 2.0, 'per_ls': 1.0,
            'alpha': 1e-2
        }
    
    # Phase 2: Try anisotropic length scales if d > 1 and time permits
    if d > 1 and time.time() < deadline - 60:
        base_ls = best_params['rbf_ls'][0]
        # Try varying length scales per dimension
        for dim in range(d):
            if time.time() > deadline:
                break
            for ls_mult in [0.5, 0.7, 1.3, 2.0]:
                if time.time() > deadline:
                    break
                new_ls = best_params['rbf_ls'].copy()
                new_ls[dim] = base_ls * ls_mult
                params = best_params.copy()
                params['rbf_ls'] = new_ls
                
                cv_mse = _cv_score(X_norm, y_centered, params, params['alpha'])
                if cv_mse < best_cv_mse:
                    best_cv_mse = cv_mse
                    best_params = params.copy()
    
    # Phase 3: Try adding polynomial kernel
    poly_configs = [
        {'weight': 0.1, 'degree': 2, 'gamma': 1.0, 'coef0': 1.0},
        {'weight': 0.5, 'degree': 2, 'gamma': 1.0, 'coef0': 1.0},
        {'weight': 1.0, 'degree': 2, 'gamma': 1.0, 'coef0': 1.0},
        {'weight': 0.5, 'degree': 3, 'gamma': 0.5, 'coef0': 1.0},
        {'weight': 0.5, 'degree': 4, 'gamma': 0.5, 'coef0': 1.0},
    ]
    
    for poly_cfg in poly_configs:
        if time.time() > deadline:
            break
        for ls_mult in [0.5, 1.0, 2.0]:
            if time.time() > deadline:
                break
            for alpha_mult in [0.1, 1.0, 10.0]:
                if time.time() > deadline:
                    break
                
                ls_val = best_params['rbf_ls'].copy() * ls_mult
                alpha = best_params['alpha'] * alpha_mult
                
                params = best_params.copy()
                params['rbf_ls'] = ls_val
                params['alpha'] = alpha
                params['poly_weight'] = poly_cfg['weight']
                params['poly_degree'] = poly_cfg['degree']
                params['poly_gamma'] = poly_cfg['gamma']
                params['poly_coef0'] = poly_cfg['coef0']
                
                cv_mse = _cv_score(X_norm, y_centered, params, alpha)
                if cv_mse < best_cv_mse:
                    best_cv_mse = cv_mse
                    best_params = params.copy()
    
    # Phase 4: Try adding periodic kernel
    per_configs = [
        {'weight': 0.1, 'period': 2.0, 'ls': 0.5},
        {'weight': 0.5, 'period': 2.0, 'ls': 0.5},
        {'weight': 0.1, 'period': 1.0, 'ls': 0.3},
        {'weight': 0.5, 'period': 1.0, 'ls': 0.3},
        {'weight': 0.1, 'period': 5.0, 'ls': 1.0},
        {'weight': 0.5, 'period': 5.0, 'ls': 1.0},
    ]
    
    for per_cfg in per_configs:
        if time.time() > deadline:
            break
        
        params = best_params.copy()
        params['per_weight'] = per_cfg['weight']
        params['per_period'] = per_cfg['period']
        params['per_ls'] = per_cfg['ls']
        
        cv_mse = _cv_score(X_norm, y_centered, params, best_params['alpha'])
        if cv_mse < best_cv_mse:
            best_cv_mse = cv_mse
            best_params = params.copy()
    
    # Phase 5: Fine-tune around best parameters
    if time.time() < deadline - 30:
        base_ls = best_params['rbf_ls'].copy()
        base_alpha = best_params['alpha']
        
        for ls_mult in [0.7, 0.8, 0.9, 1.1, 1.2, 1.3]:
            if time.time() > deadline:
                break
            for alpha_mult in [0.5, 0.7, 1.3, 2.0]:
                if time.time() > deadline:
                    break
                
                params = best_params.copy()
                params['rbf_ls'] = base_ls * ls_mult
                params['alpha'] = base_alpha * alpha_mult
                
                cv_mse = _cv_score(X_norm, y_centered, params, params['alpha'])
                if cv_mse < best_cv_mse:
                    best_cv_mse = cv_mse
                    best_params = params.copy()
    
    # Final fit with best parameters
    final_params = best_params
    K_final = _compute_kernel(X_norm, X_norm, final_params)
    coeffs_final = _kernel_ridge_solve(K_final, y_centered, final_params['alpha'])
    
    def predict(X):
        X_test_norm = (X - x_min) / x_range
        K_test = _compute_kernel(X_test_norm, X_norm, final_params)
        return K_test @ coeffs_final + y_mean
    
    return predict
# EVOLVE-BLOCK-END
