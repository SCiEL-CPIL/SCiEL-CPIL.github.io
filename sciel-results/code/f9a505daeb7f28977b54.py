"""Symbolic regression via Adaptive Kernel Ridge Regression (Ensemble) with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import operator
from scipy.optimize import minimize

def fit(X_train, y_train):
    start_time = time.time()
    # 300s limit. We reserve 30s for safety/overhead, leaving 270s for computation.
    deadline = start_time + 270.0
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64).flatten()
    n_samples, n_features = X.shape
    
    # Normalize inputs to [0, 1] for numerical stability in distance calculations
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0
    X_norm = (X - X_min) / X_range
    
    # Standardize y
    y_mean = y.mean()
    y_std = y.std()
    if y_std == 0:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std

    # Strategy: Kernel Ridge Regression (KRR) Ensemble
    # We use an ensemble of kernels to capture different structures:
    # 1. RBF (Gaussian): Captures smooth, local variations.
    # 2. Polynomial: Captures global polynomial trends.
    # 3. Sine: Captures periodic behaviors.
    
    # Pre-compute kernels for the training set for each candidate hyperparameter set
    # To save time, we will optimize hyperparameters for a few representative kernels
    # and then combine them.
    
    # Helper to compute RBF kernel
    def rbf_kernel(X1, X2, length_scale):
        # X1: (n, d), X2: (m, d)
        # sq_dists = sum((X1 - X2)^2)
        # Efficiently: ||a-b||^2 = ||a||^2 - 2 a.b + ||b||^2
        X1_sq = np.sum(X1**2, axis=1)
        X2_sq = np.sum(X2**2, axis=1)
        sq_dists = X1_sq[:, None] + X2_sq[None, :] - 2 * np.dot(X1, X2.T)
        sq_dists = np.maximum(sq_dists, 0) # Numerical stability
        return np.exp(-sq_dists / (2 * length_scale**2))

    def poly_kernel(X1, X2, degree, gamma):
        return (gamma * np.dot(X1, X2.T) + 1) ** degree

    def sine_kernel(X1, X2, length_scale):
        # Sine kernel is stationary: k(x, y) = sin(||x-y|| / l) / (||x-y|| / l)
        # This is equivalent to a sinc function.
        X1_sq = np.sum(X1**2, axis=1)
        X2_sq = np.sum(X2**2, axis=1)
        sq_dists = X1_sq[:, None] + X2_sq[None, :] - 2 * np.dot(X1, X2.T)
        sq_dists = np.maximum(sq_dists, 0)
        dists = np.sqrt(sq_dists)
        scaled_dists = dists / length_scale
        # Handle 0/0
        kernel_vals = np.ones_like(scaled_dists)
        mask = scaled_dists > 1e-8
        kernel_vals[mask] = np.sin(scaled_dists[mask]) / scaled_dists[mask]
        return kernel_vals

    # We will try a few configurations for each kernel type.
    # Since n_samples is small (200), solving KRR is O(N^3) ~ 8M ops, which is fast.
    # We can afford to try a grid of hyperparameters.
    
    models = []
    
    # 1. RBF Kernel with varying length scales
    # Length scale in normalized [0,1] space. Small l -> local, large l -> global.
    rbf_ls_candidates = [0.1, 0.2, 0.5, 1.0, 2.0]
    for ls in rbf_ls_candidates:
        if time.time() > deadline - 20:
            break
        try:
            K = rbf_kernel(X_norm, X_norm, ls)
            # Add small regularization for stability
            reg = 1e-5
            K_reg = K + reg * np.eye(n_samples)
            alpha = np.linalg.solve(K_reg, y_norm)
            
            # Evaluate training error (proxy for test error in noiseless case)
            pred_train = K @ alpha
            mse = np.mean((pred_train - y_norm)**2)
            
            # Store model
            models.append({
                'type': 'rbf',
                'ls': ls,
                'alpha': alpha,
                'mse': mse
            })
        except:
            pass

    # 2. Polynomial Kernel
    poly_degrees = [2, 3]
    poly_gammas = [1.0, 2.0]
    for deg in poly_degrees:
        for gamma in poly_gammas:
            if time.time() > deadline - 20:
                break
            try:
                K = poly_kernel(X_norm, X_norm, deg, gamma)
                reg = 1e-5
                K_reg = K + reg * np.eye(n_samples)
                alpha = np.linalg.solve(K_reg, y_norm)
                
                pred_train = K @ alpha
                mse = np.mean((pred_train - y_norm)**2)
                
                models.append({
                    'type': 'poly',
                    'deg': deg,
                    'gamma': gamma,
                    'alpha': alpha,
                    'mse': mse
                })
            except:
                pass

    # 3. Sine Kernel
    sine_ls_candidates = [0.5, 1.0, 2.0]
    for ls in sine_ls_candidates:
        if time.time() > deadline - 20:
            break
        try:
            K = sine_kernel(X_norm, X_norm, ls)
            reg = 1e-5
            K_reg = K + reg * np.eye(n_samples)
            alpha = np.linalg.solve(K_reg, y_norm)
            
            pred_train = K @ alpha
            mse = np.mean((pred_train - y_norm)**2)
            
            models.append({
                'type': 'sine',
                'ls': ls,
                'alpha': alpha,
                'mse': mse
            })
        except:
            pass

    # If no models were successfully created, fallback to simple linear regression
    if not models:
        try:
            # Add bias term
            X_bias = np.column_stack([X_norm, np.ones(n_samples)])
            w, *_ = np.linalg.lstsq(X_bias, y_norm, rcond=None)
            def predict(X_test):
                X_test = np.asarray(X_test, dtype=np.float64)
                X_test_norm = (X_test - X_min) / X_range
                X_test_bias = np.column_stack([X_test_norm, np.ones(X_test_norm.shape[0])])
                pred_norm = X_test_bias @ w
                return pred_norm * y_std + y_mean
            return predict
        except:
            def predict(X_test):
                return np.zeros(len(X_test)) * y_mean
            return predict

    # Select the best model based on training MSE
    # In noiseless data, the model with the lowest training MSE is likely the best.
    # However, sometimes a slightly regularized model generalizes better if the true function
    # is simple. But with only 200 points and smooth functions, the best fit is usually correct.
    # We'll pick the one with the lowest MSE.
    
    best_model = min(models, key=lambda m: m['mse'])
    
    # Optional: Ensemble the top 3 models if they have similar performance
    # For now, stick to the single best to avoid dilution if one is clearly superior.
    
    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        X_test_norm = (X_test - X_min) / X_range
        
        # Compute kernel matrix between test and train
        if best_model['type'] == 'rbf':
            K_test = rbf_kernel(X_test_norm, X_norm, best_model['ls'])
        elif best_model['type'] == 'poly':
            K_test = poly_kernel(X_test_norm, X_norm, best_model['deg'], best_model['gamma'])
        elif best_model['type'] == 'sine':
            K_test = sine_kernel(X_test_norm, X_norm, best_model['ls'])
        else:
            # Fallback
            return np.zeros(X_test_norm.shape[0]) * y_mean
            
        pred_norm = K_test @ best_model['alpha']
        pred = pred_norm * y_std + y_mean
        
        # Ensure finite
        pred = np.nan_to_num(pred, nan=y_mean, posinf=y_mean, neginf=y_mean)
        
        return pred

    return predict
# EVOLVE-BLOCK-END
