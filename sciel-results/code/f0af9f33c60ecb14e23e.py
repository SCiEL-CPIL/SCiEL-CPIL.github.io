"""
Strategy: Global Fourier series regression with adaptive frequency cutoff.
Instead of local kernel regression, we represent the function as a sum of sines
and cosines (a global periodic basis). We search over the number of harmonics
(M) using cross-validation to balance bias and variance. This leverages the
smoothness of the hidden functions and can capture oscillatory components that
local methods might miss or handle suboptimally. The design matrix is built
once per M and solved via least squares, which is fast and stable.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import lstsq

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280  # Leave 20s buffer for 300s limit
    
    X = X_train[:, 0]  # 1D input
    y = y_train
    
    n = len(X)
    
    # Determine the range of the input to scale for Fourier basis
    x_min = np.min(X)
    x_max = np.max(X)
    x_range = x_max - x_min
    
    if x_range < 1e-10:
        # Degenerate case: all inputs are the same
        mean_y = np.mean(y)
        def predict(X, mean_y=mean_y):
            return np.full(len(X.flatten()), mean_y).reshape(X.shape)
        return predict
    
    # We will search over the number of harmonics M
    # M=1 means only the first harmonic (sin, cos), etc.
    # We also include a constant term (DC component) implicitly or explicitly?
    # Standard Fourier series: a0 + sum_{k=1}^M (a_k cos(k*w*x) + b_k sin(k*w*x))
    # Here w = 2*pi / (x_max - x_min) ? Or just use the natural frequency?
    # Since the domain is not necessarily periodic, we map x to [0, 2*pi] or similar.
    # Let's map x in [x_min, x_max] to t in [0, 2*pi].
    
    def transform_x(x):
        # Map x to [0, 2*pi]
        return 2 * np.pi * (x - x_min) / x_range
    
    # Search over M (number of harmonics)
    # M=0: just constant
    # M=1: constant + 1st harmonic
    # ...
    # We'll try M from 1 to 50 or so, depending on time.
    
    best_score = np.inf
    best_M = 1
    best_coeffs = None
    
    # Cross-validation setup
    n_folds = 5
    fold_size = n // n_folds
    
    # Pre-compute transformed X
    X_t = transform_x(X)
    
    # We will try a range of M values
    # Since n=200, we can have up to ~100 harmonics before overfitting, but CV will pick.
    # Let's try M from 1 to 60.
    M_max = 60
    
    for M in range(1, M_max + 1):
        if time.time() > deadline:
            break
        
        try:
            cv_errors = []
            
            for i in range(n_folds):
                val_start = i * fold_size
                val_end = (i + 1) * fold_size if i < n_folds - 1 else n
                val_idx = np.arange(val_start, val_end)
                train_idx = np.concatenate([np.arange(0, val_start), np.arange(val_end, n)])
                
                X_val_t = X_t[val_idx]
                y_val = y[val_idx]
                X_tr_t = X_t[train_idx]
                y_tr = y[train_idx]
                
                # Build design matrix for training
                # Columns: [1, cos(t), sin(t), cos(2t), sin(2t), ...]
                # Total columns = 1 + 2*M
                A_tr = np.ones((len(X_tr_t), 1 + 2 * M))
                for k in range(1, M + 1):
                    A_tr[:, 1 + 2*(k-1)] = np.cos(k * X_tr_t)
                    A_tr[:, 2 + 2*(k-1)] = np.sin(k * X_tr_t)
                
                # Solve least squares
                # Use lstsq for stability
                coeffs, residuals, rank, s = lstsq(A_tr, y_tr)
                
                # Build design matrix for validation
                A_val = np.ones((len(X_val_t), 1 + 2 * M))
                for k in range(1, M + 1):
                    A_val[:, 1 + 2*(k-1)] = np.cos(k * X_val_t)
                    A_val[:, 2 + 2*(k-1)] = np.sin(k * X_val_t)
                
                # Predict
                pred = A_val @ coeffs
                
                if not np.all(np.isfinite(pred)):
                    cv_errors.append(np.inf)
                else:
                    mse = np.mean((pred - y_val) ** 2)
                    cv_errors.append(mse)
            
            mean_mse = np.mean(cv_errors)
            
            if mean_mse < best_score:
                best_score = mean_mse
                best_M = M
                
        except Exception:
            continue
    
    # Refine the search around the best M found (optional, but good for precision)
    # Let's do a second pass with finer granularity if time permits
    if time.time() < deadline - 10:
        M_start = max(1, best_M - 5)
        M_end = min(M_max, best_M + 5)
        
        for M in range(M_start, M_end + 1):
            if time.time() > deadline:
                break
            
            try:
                cv_errors = []
                
                for i in range(n_folds):
                    val_start = i * fold_size
                    val_end = (i + 1) * fold_size if i < n_folds - 1 else n
                    val_idx = np.arange(val_start, val_end)
                    train_idx = np.concatenate([np.arange(0, val_start), np.arange(val_end, n)])
                    
                    X_val_t = X_t[val_idx]
                    y_val = y[val_idx]
                    X_tr_t = X_t[train_idx]
                    y_tr = y[train_idx]
                    
                    A_tr = np.ones((len(X_tr_t), 1 + 2 * M))
                    for k in range(1, M + 1):
                        A_tr[:, 1 + 2*(k-1)] = np.cos(k * X_tr_t)
                        A_tr[:, 2 + 2*(k-1)] = np.sin(k * X_tr_t)
                    
                    coeffs, residuals, rank, s = lstsq(A_tr, y_tr)
                    
                    A_val = np.ones((len(X_val_t), 1 + 2 * M))
                    for k in range(1, M + 1):
                        A_val[:, 1 + 2*(k-1)] = np.cos(k * X_val_t)
                        A_val[:, 2 + 2*(k-1)] = np.sin(k * X_val_t)
                    
                    pred = A_val @ coeffs
                    
                    if not np.all(np.isfinite(pred)):
                        cv_errors.append(np.inf)
                    else:
                        mse = np.mean((pred - y_val) ** 2)
                        cv_errors.append(mse)
                
                mean_mse = np.mean(cv_errors)
                
                if mean_mse < best_score:
                    best_score = mean_mse
                    best_M = M
                    
            except Exception:
                continue
    
    # Fit the final model on all data with the best M
    M_final = best_M
    
    A_full = np.ones((n, 1 + 2 * M_final))
    for k in range(1, M_final + 1):
        A_full[:, 1 + 2*(k-1)] = np.cos(k * X_t)
        A_full[:, 2 + 2*(k-1)] = np.sin(k * X_t)
    
    coeffs_final, residuals, rank, s = lstsq(A_full, y)
    
    mean_y = np.mean(y)
    
    def predict(X, A_full=A_full, coeffs_final=coeffs_final, M_final=M_final, 
                X_t=X_t, x_min=x_min, x_range=x_range, mean_y=mean_y):
        X_flat = X.flatten()
        
        # Transform inputs
        X_flat_t = 2 * np.pi * (X_flat - x_min) / x_range
        
        # Build design matrix
        A_pred = np.ones((len(X_flat_t), 1 + 2 * M_final))
        for k in range(1, M_final + 1):
            A_pred[:, 1 + 2*(k-1)] = np.cos(k * X_flat_t)
            A_pred[:, 2 + 2*(k-1)] = np.sin(k * X_flat_t)
        
        # Predict
        pred = A_pred @ coeffs_final
        
        # Ensure finite
        pred = np.where(np.isfinite(pred), pred, mean_y)
        
        return pred.reshape(X.shape)
    
    return predict
# EVOLVE-BLOCK-END
