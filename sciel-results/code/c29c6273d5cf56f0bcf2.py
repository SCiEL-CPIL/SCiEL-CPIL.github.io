"""Gaussian Process regression with Matérn-5/2 kernel and grid-search hyperparameters."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def fit(X_train, y_train):
    # X_train shape (n, d), y_train shape (n,)
    # All equations are 1D, so d=1
    n = X_train.shape[0]
    x = X_train[:, 0].copy()
    y = y_train.copy()
    
    # Sort by x for potential efficiency, though GP doesn't strictly require it
    # But sorting can help with numerical stability in some cases
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Center and scale x for numerical stability
    x_mean = np.mean(x_sorted)
    x_std = np.std(x_sorted)
    if x_std < 1e-10:
        x_std = 1.0
    x_scaled = (x_sorted - x_mean) / x_std
    
    # Center y for numerical stability
    y_mean = np.mean(y_sorted)
    y_std = np.std(y_sorted)
    if y_std < 1e-10:
        y_std = 1.0
    y_scaled = (y_sorted - y_mean) / y_std
    
    def matern52_kernel(x1, x2, length_scale, signal_var):
        """Matérn 5/2 kernel for 1D inputs (already scaled)."""
        diff = x1[:, None] - x2[None, :]
        z = np.abs(diff) / length_scale
        # Matérn 5/2: (1 + sqrt(5)*z + 5/3*z^2) * exp(-sqrt(5)*z)
        kernel = (1.0 + np.sqrt(5.0) * z + (5.0 / 3.0) * z**2) * np.exp(-np.sqrt(5.0) * z)
        return signal_var * kernel
    
    def log_likelihood(params, x_data, y_data, alpha=1e-6):
        """Negative log marginal likelihood."""
        length_scale = np.exp(params[0])  # ensure positive
        signal_var = np.exp(params[1])    # ensure positive
        noise_var = np.exp(params[2])     # ensure positive
        
        K = matern52_kernel(x_data, x_data, length_scale, signal_var) + noise_var * np.eye(len(x_data))
        
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # Add jitter if Cholesky fails
            K += 1e-8 * np.eye(len(x_data))
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                return 1e10  # very large penalty
        
        alpha_vec = np.linalg.solve(L.T, np.linalg.solve(L, y_data))
        nll = 0.5 * y_data @ alpha_vec + np.sum(np.log(np.diag(L))) + 0.5 * len(y_data) * np.log(2 * np.pi)
        return nll
    
    # Grid search over hyperparameters using 5-fold cross-validation
    # Since n=200, we can do a moderate grid search
    
    # Define grid for length_scale (on log scale) and signal_var (on log scale)
    # We'll work in scaled x space, so length_scale is in scaled units
    ls_grid = np.exp(np.linspace(-2, 2, 15))  # length scales from ~0.13 to ~7.4
    sv_grid = np.exp(np.linspace(-2, 2, 15))  # signal variances from ~0.13 to ~7.4
    nv_grid = np.exp(np.linspace(-5, -1, 8))  # noise variances from ~0.0067 to ~0.37
    
    # 5-fold cross-validation
    n_folds = 5
    fold_size = n // n_folds
    indices = np.arange(n)
    
    # Shuffle indices for CV
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    
    best_nll = np.inf
    best_params = None
    
    # For efficiency, precompute fold indices
    fold_indices = []
    for i in range(n_folds):
        val_idx = perm[i * fold_size : (i + 1) * fold_size]
        train_idx = np.setdiff1d(perm, val_idx)
        fold_indices.append((train_idx, val_idx))
    
    # Grid search
    for ls in ls_grid:
        for sv in sv_grid:
            for nv in nv_grid:
                total_nll = 0.0
                for train_idx, val_idx in fold_indices:
                    x_tr = x_scaled[train_idx]
                    y_tr = y_scaled[train_idx]
                    x_va = x_scaled[val_idx]
                    y_va = y_scaled[val_idx]
                    
                    K_train = matern52_kernel(x_tr, x_tr, ls, sv) + nv * np.eye(len(x_tr))
                    K_test = matern52_kernel(x_va, x_tr, ls, sv)
                    
                    try:
                        L = np.linalg.cholesky(K_train)
                    except np.linalg.LinAlgError:
                        K_train += 1e-8 * np.eye(len(x_tr))
                        try:
                            L = np.linalg.cholesky(K_train)
                        except np.linalg.LinAlgError:
                            total_nll = np.inf
                            break
                    
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_tr))
                    y_pred = K_test @ alpha
                    residual = y_va - y_pred
                    nll_fold = 0.5 * np.sum(residual**2) + len(val_idx) * 0.5 * np.log(2 * np.pi * nv)
                    total_nll += nll_fold
                
                if total_nll < best_nll:
                    best_nll = total_nll
                    best_params = (ls, sv, nv)
    
    # Extract best hyperparameters
    best_ls, best_sv, best_nv = best_params
    
    # Final GP fit using all training data with best hyperparameters
    K = matern52_kernel(x_scaled, x_scaled, best_ls, best_sv) + best_nv * np.eye(n)
    
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        K += 1e-8 * np.eye(n)
        L = np.linalg.cholesky(K)
    
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_scaled))
    
    # Create predict function
    def predict(X):
        x_new = X[:, 0].copy()
        x_new_scaled = (x_new - x_mean) / x_std
        
        K_star = matern52_kernel(x_new_scaled, x_scaled, best_ls, best_sv)
        y_pred_scaled = K_star @ alpha
        y_pred = y_pred_scaled * y_std + y_mean
        return y_pred
    
    return predict
# EVOLVE-BLOCK-END
