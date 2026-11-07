"""Symbolic regression with kernel ensemble (RBF, Poly, Linear)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global time tracker
_total_start = None
_time_limit = 320.0  # Leave 10s safety margin before 330s kill

def _elapsed():
    global _total_start
    if _total_start is None:
        _total_start = time.time()
    return time.time() - _total_start

def _remaining():
    return _time_limit - _elapsed()

def _rbf_kernel(X1, X2, gamma):
    """Compute RBF kernel matrix between X1 and X2."""
    sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
    np.maximum(sq_dists, 0, out=sq_dists)
    return np.exp(-gamma * sq_dists)

def _poly_kernel(X1, X2, degree, coef0):
    """Compute polynomial kernel matrix between X1 and X2."""
    return (X1 @ X2.T + coef0) ** degree

def _linear_kernel(X1, X2):
    """Compute linear kernel matrix between X1 and X2."""
    return X1 @ X2.T

def fit(X_train, y_train):
    global _total_start
    if _total_start is None:
        _total_start = time.time()
    
    n, d = X_train.shape
    X = X_train.astype(np.float64)
    y = y_train.astype(np.float64)
    
    # Normalize inputs for better kernel behavior
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-10] = 1.0
    X_norm = (X - X_mean) / X_std
    
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-10:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    # Split for validation
    idx = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(idx)
    val_size = max(20, int(0.1 * n))
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    
    X_tr = X_norm[train_idx]
    y_tr = y_norm[train_idx]
    X_val = X_norm[val_idx]
    y_val = y_norm[val_idx]
    
    # Define kernel configurations to try
    # Each config: (kernel_type, params)
    # We'll search over a grid of parameters for each kernel type
    
    # RBF kernels with different gammas
    rbf_gammas = [1.0 / (d * np.var(X_norm)), 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
    # Polynomial kernels
    poly_configs = [(2, 1.0), (2, 0.5), (3, 1.0), (2, 2.0), (3, 0.5)]
    
    # Ridge regularization for kernel methods
    ridge_lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]
    
    best_val_mse = np.inf
    best_config = None  # (kernel_type, kernel_params, lambda)
    best_alpha = None
    best_kernel_matrix_train = None
    
    # Try each kernel configuration
    # RBF kernels
    for gamma in rbf_gammas:
        if _remaining() < 5.0:
            break
        try:
            K_tr = _rbf_kernel(X_tr, X_tr, gamma)
            K_val_tr = _rbf_kernel(X_val, X_tr, gamma)
            
            for lam in ridge_lambdas:
                try:
                    A = K_tr + lam * np.eye(len(X_tr))
                    alpha = np.linalg.solve(A, y_tr)
                    pred_val = K_val_tr @ alpha
                    mse = np.mean((pred_val - y_val) ** 2)
                    if mse < best_val_mse:
                        best_val_mse = mse
                        best_config = ('rbf', gamma, lam)
                        best_alpha = alpha
                        best_kernel_matrix_train = K_tr
                except np.linalg.LinAlgError:
                    continue
        except Exception:
            continue
    
    # Polynomial kernels
    for degree, coef0 in poly_configs:
        if _remaining() < 5.0:
            break
        try:
            K_tr = _poly_kernel(X_tr, X_tr, degree, coef0)
            K_val_tr = _poly_kernel(X_val, X_tr, degree, coef0)
            
            for lam in ridge_lambdas:
                try:
                    A = K_tr + lam * np.eye(len(X_tr))
                    alpha = np.linalg.solve(A, y_tr)
                    pred_val = K_val_tr @ alpha
                    mse = np.mean((pred_val - y_val) ** 2)
                    if mse < best_val_mse:
                        best_val_mse = mse
                        best_config = ('poly', (degree, coef0), lam)
                        best_alpha = alpha
                        best_kernel_matrix_train = K_tr
                except np.linalg.LinAlgError:
                    continue
        except Exception:
            continue
    
    # Linear kernel (just in case)
    if _remaining() > 5.0:
        try:
            K_tr = _linear_kernel(X_tr, X_tr)
            K_val_tr = _linear_kernel(X_val, X_tr)
            
            for lam in ridge_lambdas:
                try:
                    A = K_tr + lam * np.eye(len(X_tr))
                    alpha = np.linalg.solve(A, y_tr)
                    pred_val = K_val_tr @ alpha
                    mse = np.mean((pred_val - y_val) ** 2)
                    if mse < best_val_mse:
                        best_val_mse = mse
                        best_config = ('linear', None, lam)
                        best_alpha = alpha
                        best_kernel_matrix_train = K_tr
                except np.linalg.LinAlgError:
                    continue
        except Exception:
            pass
    
    # If no kernel config worked well, fall back to simple linear regression
    if best_config is None or best_val_mse > 1.0:
        # Use linear model as fallback
        try:
            w, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(X_tr)), X_tr]), y_tr, rcond=None)
            pred_val = np.column_stack([np.ones(len(X_val)), X_val]) @ w
            mse = np.mean((pred_val - y_val) ** 2)
            if mse < best_val_mse:
                best_val_mse = mse
                best_config = ('fallback_linear', None, None)
                best_alpha = w
                best_kernel_matrix_train = None
        except Exception:
            pass
    
    # Final fit on ALL data with the best configuration
    if best_config is not None and best_config[0] != 'fallback_linear':
        kernel_type = best_config[0]
        kernel_params = best_config[1]
        lam = best_config[2]
        
        try:
            if kernel_type == 'rbf':
                K_full = _rbf_kernel(X_norm, X_norm, kernel_params)
            elif kernel_type == 'poly':
                degree, coef0 = kernel_params
                K_full = _poly_kernel(X_norm, X_norm, degree, coef0)
            elif kernel_type == 'linear':
                K_full = _linear_kernel(X_norm, X_norm)
            else:
                K_full = None
            
            if K_full is not None:
                A = K_full + lam * np.eye(n)
                final_alpha = np.linalg.solve(A, y_norm)
            else:
                final_alpha = best_alpha
        except np.linalg.LinAlgError:
            final_alpha = best_alpha
    else:
        # Fallback: fit linear on all data
        try:
            final_alpha, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), X_norm]), y_norm, rcond=None)
        except Exception:
            final_alpha = np.zeros(d + 1)
    
    # Determine prediction method
    use_fallback_linear = (best_config is not None and best_config[0] == 'fallback_linear')
    
    if use_fallback_linear:
        def predict(X_test):
            X_t = X_test.astype(np.float64)
            X_t_norm = (X_t - X_mean) / X_std
            X_t_aug = np.column_stack([np.ones(len(X_t)), X_t_norm])
            y_pred_norm = X_t_aug @ final_alpha
            return y_pred_norm * y_std + y_mean
    else:
        kernel_type = best_config[0]
        kernel_params = best_config[1]
        
        def predict(X_test):
            X_t = X_test.astype(np.float64)
            X_t_norm = (X_t - X_mean) / X_std
            
            if kernel_type == 'rbf':
                K_test = _rbf_kernel(X_t_norm, X_norm, kernel_params)
            elif kernel_type == 'poly':
                degree, coef0 = kernel_params
                K_test = _poly_kernel(X_t_norm, X_norm, degree, coef0)
            elif kernel_type == 'linear':
                K_test = _linear_kernel(X_t_norm, X_norm)
            else:
                # Should not reach here
                K_test = _linear_kernel(X_t_norm, X_norm)
            
            y_pred_norm = K_test @ final_alpha
            return y_pred_norm * y_std + y_mean
    
    return predict
# EVOLVE-BLOCK-END
