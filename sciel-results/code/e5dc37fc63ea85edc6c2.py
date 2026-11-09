"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280  # Safety margin for 300s limit
    
    X = X_train
    y = y_train
    n = len(y)
    
    # Scale X to [-1, 1] for better kernel conditioning
    # Using min/max from training data
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    x_range = x_max - x_min
    # Avoid division by zero if all inputs are same (unlikely but safe)
    x_range[x_range == 0] = 1.0
    
    X_scaled = (X - x_min) / x_range * 2 - 1
    
    # Standardize y for numerical stability in GP? 
    # GP is invariant to linear transformations of y if we transform back, 
    # but scaling y helps with sigma_f initialization.
    y_mean = y.mean()
    y_std = y.std()
    if y_std == 0:
        y_std = 1.0
    y_scaled = (y - y_mean) / y_std
    
    def matern52_kernel(X1, X2, ell, sigma_f):
        # X1, X2 are (n, 1) arrays
        D = cdist(X1, X2)
        sqrt5D = np.sqrt(5.0) * D / ell
        K = (sigma_f ** 2) * (1.0 + sqrt5D + (5.0 * D ** 2) / (3.0 * ell ** 2)) * np.exp(-sqrt5D)
        return K
    
    def neg_log_marginal_likelihood(params):
        log_ell = params[0]
        log_sigma_f = params[1]
        ell = np.exp(log_ell)
        sigma_f = np.exp(log_sigma_f)
        
        # Add small noise for numerical stability
        noise = 1e-6 * (y_std ** 2) 
        # Note: y_scaled has std 1, so noise should be small relative to 1.
        # Let's use absolute noise on scaled y.
        noise_val = 1e-6 
        
        K = matern52_kernel(X_scaled, X_scaled, ell, sigma_f)
        K[np.diag_indices_from(K)] += noise_val
        
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10 # Large penalty for singular matrix
            
        # Compute log likelihood
        # log p(y|X) = -0.5 * y^T K^-1 y - 0.5 log det(K) - n/2 log(2pi)
        # log det(K) = 2 sum log(diag(L))
        # K^-1 y = L^-T L^-1 y
        
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_scaled))
        log_det_K = 2.0 * np.sum(np.log(np.diag(L)))
        
        nll = 0.5 * (y_scaled @ alpha) + 0.5 * log_det_K + 0.5 * n * np.log(2.0 * np.pi)
        
        if not np.isfinite(nll):
            return 1e10
            
        return nll

    # Initial guesses
    # Median distance between points
    if n > 1:
        D_train = cdist(X_scaled, X_scaled)
        np.fill_diagonal(D_train, np.inf)
        init_ell = np.median(D_train)
        if init_ell < 1e-6:
            init_ell = 1.0
    else:
        init_ell = 1.0
        
    init_sigma_f = 1.0 # Since y is scaled to std 1
    
    x0 = [np.log(init_ell), np.log(init_sigma_f)]
    
    best_params = x0
    best_nll = np.inf
    
    try:
        # Use Nelder-Mead for robustness with small parameter space
        res = minimize(neg_log_marginal_likelihood, x0, method='Nelder-Mead', 
                       options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if res.success or res.fun < best_nll:
            if np.isfinite(res.fun):
                best_params = res.x
                best_nll = res.fun
    except Exception:
        pass

    # If optimization failed or didn't improve, use initial guess
    log_ell, log_sigma_f = best_params
    ell = np.exp(log_ell)
    sigma_f = np.exp(log_sigma_f)
    
    noise_val = 1e-6
    
    K = matern52_kernel(X_scaled, X_scaled, ell, sigma_f)
    K[np.diag_indices_from(K)] += noise_val
    
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_scaled))
    except np.linalg.LinAlgError:
        # Fallback to linear regression if GP fails
        def predict_fallback(X_test):
            X_test_scaled = (X_test - x_min) / x_range * 2 - 1
            # Simple linear fit
            A = np.column_stack([np.ones(len(X_test_scaled)), X_test_scaled])
            w, *_ = np.linalg.lstsq(A, y, rcond=None)
            return A @ w
        return predict_fallback

    def predict(X_test):
        if time.time() > deadline:
             # Emergency fallback if we are running out of time during prediction?
             # Prediction is fast, but let's be safe.
             pass
             
        X_test_scaled = (X_test - x_min) / x_range * 2 - 1
        K_s = matern52_kernel(X_test_scaled, X_scaled, ell, sigma_f)
        
        # Prediction mean
        # y_pred_scaled = K_s @ alpha
        y_pred_scaled = K_s @ alpha
        y_pred = y_pred_scaled * y_std + y_mean
        
        # Ensure finite values
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
        
        return y_pred

    return predict

# EVOLVE-BLOCK-END
