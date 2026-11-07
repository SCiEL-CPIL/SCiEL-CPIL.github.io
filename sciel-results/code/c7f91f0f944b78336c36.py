import numpy as np
import time
from scipy.interpolate import PchipInterpolator

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 300  # 300 seconds total limit for all 4 equations, assume we get ~75s each roughly, but we should be fast anyway.
    
    # Flatten and sort
    X_flat = np.asarray(X_train).flatten()
    y_flat = np.asarray(y_train).flatten()
    
    # Handle potential duplicates by averaging
    unique_X, inverse_indices, counts = np.unique(X_flat, return_inverse=True, return_counts=True)
    if len(unique_X) != len(X_flat):
        y_sum = np.zeros_like(unique_X, dtype=float)
        np.add.at(y_sum, inverse_indices, y_flat)
        y_flat = y_sum / counts
        X_flat = unique_X
        
    # If very few points, return mean
    if len(X_flat) < 2:
        mean_y = np.mean(y_flat)
        return lambda X: np.full_like(np.asarray(X), mean_y, dtype=float)

    N = len(X_flat)
    
    # Try PCHIP first (monotone cubic interpolation) - ideal for noiseless 1D data
    try:
        # Sort by X
        sort_idx = np.argsort(X_flat)
        X_sorted = X_flat[sort_idx]
        y_sorted = y_flat[sort_idx]
        
        # Check for duplicate X values after sorting (shouldn't happen after unique, but be safe)
        # PCHIP requires strictly increasing x
        if np.any(np.diff(X_sorted) <= 0):
            raise ValueError("Non-strictly increasing X after sort")
            
        pchip_interp = PchipInterpolator(X_sorted, y_sorted)
        
        # Validate: check that prediction is finite
        test_pred = pchip_interp(X_sorted)
        if not np.all(np.isfinite(test_pred)):
            raise ValueError("Non-finite PCHIP predictions")
            
        def predict_pchip(X):
            X_eval = np.asarray(X).flatten()
            y_pred = pchip_interp(X_eval)
            return y_pred.reshape(np.asarray(X).shape)
            
        return predict_pchip
        
    except Exception:
        pass  # Fall through to RBF
    
    # Fallback: RBF kernel regression
    # Data scaling for kernel stability
    X_min = np.min(X_flat)
    X_max = np.max(X_flat)
    X_range = X_max - X_min
    if X_range == 0:
        X_range = 1.0
        
    # Standardize X for kernel calculation to make gamma selection easier
    X_std = (X_flat - X_min) / X_range
    
    def make_kernel(X1, X2, gamma):
        # X1, X2 are 1D arrays
        # Compute squared distances
        # (X1[:, None] - X2[None, :])**2
        diff = X1[:, None] - X2[None, :]
        return np.exp(-gamma * diff**2)

    best_w = None
    best_mse = float('inf')
    best_gamma = None
    best_lambda = None
    
    # Try different gammas (bandwidths)
    gammas = [1, 5, 10, 20, 50, 100, 200]
    
    # Try different lambdas
    lambdas = [1e-8, 1e-6, 1e-4, 1e-2, 1, 10]
    
    for gamma in gammas:
        K = make_kernel(X_std, X_std, gamma)
        
        for lam in lambdas:
            try:
                A = K + lam * np.eye(N)
                w = np.linalg.solve(A, y_flat)
                
                # Calculate training MSE
                y_pred_train = K.dot(w)
                mse = np.mean((y_pred_train - y_flat)**2)
                
                if mse < best_mse:
                    best_mse = mse
                    best_w = w
                    best_gamma = gamma
                    best_lambda = lam
                    
                # Time check: if we are taking too long, break
                if time.time() - start_time > 250:
                    break
            except np.linalg.LinAlgError:
                continue
        if time.time() - start_time > 250:
            break
            
    # If no model was found (shouldn't happen), fallback to mean
    if best_w is None:
        mean_y = np.mean(y_flat)
        return lambda X: np.full_like(np.asarray(X), mean_y, dtype=float)

    def predict(X):
        X_eval = np.asarray(X).flatten()
        # Standardize
        X_eval_std = (X_eval - X_min) / X_range
        
        # Compute kernel between test and train
        K_test = make_kernel(X_eval_std, X_std, best_gamma)
        
        # Predict
        y_pred = K_test.dot(best_w)
        
        return y_pred.reshape(np.asarray(X).shape)

    return predict
