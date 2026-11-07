import numpy as np
import time
from scipy.linalg import solve
from scipy.optimize import minimize
import warnings

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 280.0  # Safety margin from 300s limit
    
    n, d = X_train.shape
    
    # Normalize inputs to [0, 1] for kernel stability
    min_vals = X_train.min(axis=0)
    max_vals = X_train.max(axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0
    X_norm = (X_train - min_vals) / ranges
    
    # Center y
    y_mean = np.mean(y_train)
    y_centered = y_train - y_mean
    
    # Split for validation
    n_val = max(10, int(n * 0.2))
    val_idx = np.arange(n - n_val, n)
    train_idx = np.arange(0, n - n_val)
    
    X_tr_norm = X_norm[train_idx]
    y_tr = y_centered[train_idx]
    X_val_norm = X_norm[val_idx]
    y_val = y_centered[val_idx]
    
    def rbf_kernel(X1, X2, length_scale, signal_var):
        """Compute RBF kernel with given hyperparameters."""
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * (X1 @ X2.T)
        sq_dists = np.maximum(sq_dists, 0)
        K = signal_var * np.exp(-sq_dists / (2 * length_scale**2))
        if len(X1) == len(X2):
            K += 1e-8 * np.eye(len(X1))
        return K
    
    def neg_log_likelihood(params):
        """Negative log marginal likelihood for GP hyperparameters."""
        length_scale = np.exp(params[0])  # ensure positive
        signal_var = np.exp(params[1])    # ensure positive
        noise_var = np.exp(params[2])     # ensure positive
        
        K = rbf_kernel(X_tr_norm, X_tr_norm, length_scale, signal_var) + noise_var * np.eye(len(X_tr_norm))
        
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_tr))
            nll = 0.5 * y_tr @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * len(y_tr) * np.log(2 * np.pi)
            return nll
        except:
            return 1e10
    
    # Hyperparameter search for GP
    best_params = None
    best_nll = np.inf
    
    # Initial grid search for good starting points
    length_scale_grid = np.logspace(-1, 1, 15)  # 0.1 to 10
    signal_var_grid = np.logspace(-1, 2, 10)    # 0.1 to 100
    noise_var_grid = np.logspace(-4, -1, 10)    # 0.0001 to 0.1
    
    # Coarse grid search
    for ls in length_scale_grid:
        if time.time() > deadline - 60:
            break
        for sv in signal_var_grid:
            for nv in noise_var_grid:
                params = np.array([np.log(ls), np.log(sv), np.log(nv)])
                nll = neg_log_likelihood(params)
                if nll < best_nll:
                    best_nll = nll
                    best_params = params
    
    # Refine with local optimization from best grid point
    if best_params is not None:
        try:
            result = minimize(neg_log_likelihood, best_params, method='L-BFGS-B', 
                             bounds=[(-5, 5), (-2, 5), (-8, -1)], maxiter=100)
            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x
        except:
            pass
    
    # If GP failed, fall back to a simple approach
    if best_params is None or not np.isfinite(best_params).all():
        best_params = np.array([0.0, 0.0, -3.0])  # default: ls=1, sv=1, noise=0.05
    
    length_scale = np.exp(best_params[0])
    signal_var = np.exp(best_params[1])
    noise_var = np.exp(best_params[2])
    
    # Validate on holdout set
    K_val_tr = rbf_kernel(X_val_norm, X_tr_norm, length_scale, signal_var)
    K_tr_tr = rbf_kernel(X_tr_norm, X_tr_norm, length_scale, signal_var) + noise_var * np.eye(len(X_tr_norm))
    
    try:
        w = solve(K_tr_tr, y_tr, assume_a='pos')
        y_pred_val = K_val_tr @ w
        val_mse = np.mean((y_pred_val - y_val) ** 2)
    except:
        val_mse = np.inf
    
    # Compare with a simple ridge fallback to ensure GP isn't worse
    # Build simple polynomial features for fallback
    def simple_features(X):
        feats = [np.ones(X.shape[0])]
        for i in range(X.shape[1]):
            feats.append(X[:, i])
        for i in range(X.shape[1]):
            for j in range(i, X.shape[1]):
                feats.append(X[:, i] * X[:, j])
        return np.column_stack(feats)
    
    X_tr_feat = simple_features(X_tr_norm)
    X_val_feat = simple_features(X_val_norm)
    
    alpha_ridge = 1.0
    A = X_tr_feat.T @ X_tr_feat + alpha_ridge * np.eye(X_tr_feat.shape[1])
    b = X_tr_feat.T @ y_tr
    try:
        w_ridge = solve(A, b, assume_a='pos')
        y_pred_ridge = X_val_feat @ w_ridge
        ridge_mse = np.mean((y_pred_ridge - y_val) ** 2)
    except:
        ridge_mse = np.inf
    
    # Choose better model
    use_gp = val_mse < ridge_mse if np.isfinite(val_mse) and np.isfinite(ridge_mse) else (not np.isfinite(ridge_mse))
    
    if use_gp:
        # Refit GP on ALL training data
        K_full = rbf_kernel(X_norm, X_norm, length_scale, signal_var) + noise_var * np.eye(n)
        w_full = solve(K_full, y_centered, assume_a='pos')
        
        def predict(X):
            X_test_norm = (X - min_vals) / ranges
            K_test = rbf_kernel(X_test_norm, X_norm, length_scale, signal_var)
            return K_test @ w_full + y_mean
    else:
        # Refit ridge on ALL training data
        X_all_feat = simple_features(X_norm)
        A_full = X_all_feat.T @ X_all_feat + alpha_ridge * np.eye(X_all_feat.shape[1])
        b_full = X_all_feat.T @ y_centered
        w_full = solve(A_full, b_full, assume_a='pos')
        
        def predict(X):
            X_test_norm = (X - min_vals) / ranges
            X_test_feat = simple_features(X_test_norm)
            return X_test_feat @ w_full + y_mean
    
    return predict
