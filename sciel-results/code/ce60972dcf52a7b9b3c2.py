import numpy as np
import time
from scipy.optimize import minimize_scalar
import itertools

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 300.0  # Leave 30s buffer

    X = np.array(X_train, dtype=float)
    y = np.array(y_train, dtype=float)
    n, d = X.shape

    # Normalize X to unit variance to make sigma selection more robust
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    x_std[x_std < 1e-8] = 1.0
    X_norm = (X - x_mean) / x_std
    
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-8:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std

    def _time_left():
        return deadline - time.time()

    # Generate a grid of sigma values
    # Since X is normalized, sigma in [0.1, 10] is a reasonable range
    sigmas = np.logspace(-1, 1, 20)
    # Log-spaced lambda values
    lambdas = np.logspace(-4, 4, 20)

    best_score = np.inf
    best_sigma = 1.0
    best_lambda = 1.0
    best_inv_K = None

    # Pre-compute pairwise distances for efficiency? 
    # For n=200, computing K for each sigma is fast.
    
    # We will search for the best (sigma, lambda) using LOOCV
    # To save time, we can first do a coarse search, then fine-tune.
    
    def compute_loocv(sigma, lam):
        # Compute Kernel Matrix
        # K_ij = exp(-||xi - xj||^2 / (2 * sigma^2))
        # Efficiently compute squared distances
        # ||xi - xj||^2 = ||xi||^2 + ||xj||^2 - 2 * xi . xj
        X_sq = np.sum(X_norm**2, axis=1)
        sq_dists = X_sq[:, None] + X_sq[None, :] - 2 * (X_norm @ X_norm.T)
        sq_dists = np.maximum(sq_dists, 0) # Numerical safety
        
        K = np.exp(-sq_dists / (2 * sigma**2))
        
        # Add regularization
        K_reg = K + lam * np.eye(n)
        
        # Solve for weights: w = K_reg^{-1} y
        try:
            w = np.linalg.solve(K_reg, y_norm)
        except np.linalg.LinAlgError:
            return np.inf, None
            
        # Predictions
        y_pred = K @ w
        
        # LOOCV Error
        # For Ridge Regression with kernel K, the LOOCV error for point i is:
        # e_i = (y_i - y_pred_i) / (1 - H_ii)
        # where H = K (K + lam I)^{-1} K ... wait.
        # The hat matrix for kernel ridge is H = K (K + lam I)^{-1}.
        # Actually, let's just compute the diagonal of H.
        # H_ii = sum_j K_ij * w_j / y_i? No.
        # H = K (K + lam I)^{-1} K? No.
        # Model: f(x) = sum w_j K(x, x_j).
        # w = (K + lam I)^{-1} y.
        # f = K (K + lam I)^{-1} y.
        # So Hat Matrix H = K (K + lam I)^{-1}.
        
        try:
            H = K @ np.linalg.inv(K_reg) # This is expensive, better to solve
            # Better: Solve K_reg X = K for X, then H = X^T? No.
            # We want diag(H).
            # H = K @ inv(K_reg).
            # Let Z = inv(K_reg) @ K. Then H = Z.T? No.
            # Let's just compute the diagonal terms directly or use the formula:
            # LOOCV_i = (y_i - y_pred_i) / (1 - (K @ inv(K_reg))_ii)
            
            # Compute inv_K_reg once
            inv_K_reg = np.linalg.inv(K_reg)
            # H_diag = diag(K @ inv_K_reg)
            H_diag = np.einsum('ij,ji->i', K, inv_K_reg)
            
            # Avoid division by zero
            denom = 1 - H_diag
            denom[denom < 1e-8] = 1e-8
            
            loocv_errors = (y_norm - y_pred) / denom
            mse = np.mean(loocv_errors**2)
            return mse, inv_K_reg
        except:
            return np.inf, None

    # Grid Search
    # To be fast, we might skip the full grid if time is tight, but 20x20=400 iterations is small.
    # Each iteration involves 200x200 matrix inversion. 400 * O(200^3) is manageable.
    
    for sigma in sigmas:
        if _time_left() < 5:
            break
        for lam in lambdas:
            if _time_left() < 5:
                break
            mse, inv_K = compute_loocv(sigma, lam)
            if mse < best_score:
                best_score = mse
                best_sigma = sigma
                best_lambda = lam
                best_inv_K = inv_K

    # If time allows, refine around the best parameters
    if _time_left() > 30:
        # Fine-tuning
        sigma_range = np.logspace(np.log10(best_sigma) - 0.2, np.log10(best_sigma) + 0.2, 10)
        lam_range = np.logspace(np.log10(best_lambda) - 0.2, np.log10(best_lambda) + 0.2, 10)
        
        for sigma in sigma_range:
            for lam in lam_range:
                if _time_left() < 10:
                    break
                mse, inv_K = compute_loocv(sigma, lam)
                if mse < best_score:
                    best_score = mse
                    best_sigma = sigma
                    best_lambda = lam
                    best_inv_K = inv_K

    # Final Fit with best parameters
    # We already have best_inv_K, but let's recompute to be safe and get weights
    X_sq = np.sum(X_norm**2, axis=1)
    sq_dists = X_sq[:, None] + X_sq[None, :] - 2 * (X_norm @ X_norm.T)
    sq_dists = np.maximum(sq_dists, 0)
    K = np.exp(-sq_dists / (2 * best_sigma**2))
    K_reg = K + best_lambda * np.eye(n)
    
    try:
        weights = np.linalg.solve(K_reg, y_norm)
    except:
        # Fallback to simple linear regression if kernel fails
        # Or just return mean
        weights = np.zeros(n)

    def predict(X_test):
        X_test = np.array(X_test, dtype=float)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        
        # Normalize test data
        X_test_norm = (X_test - x_mean) / x_std
        
        # Compute Kernel between test and train
        X_test_sq = np.sum(X_test_norm**2, axis=1)
        X_train_sq = np.sum(X_norm**2, axis=1)
        sq_dists_test = X_test_sq[:, None] + X_train_sq[None, :] - 2 * (X_test_norm @ X_norm.T)
        sq_dists_test = np.maximum(sq_dists_test, 0)
        
        K_test = np.exp(-sq_dists_test / (2 * best_sigma**2))
        
        y_pred_norm = K_test @ weights
        y_pred = y_pred_norm * y_std + y_mean
        
        return y_pred

    return predict
