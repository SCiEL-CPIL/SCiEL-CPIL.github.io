import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def fit(X_train, y_train):
    """
    Fits a Gaussian Process regression model to the training data.
    Returns a predict callable.
    """
    start_time = time.time()
    DEADLINE = 290.0  # Safety margin before 300s limit
    
    # Ensure X is 1D for GP, though the problem states 1 input dimension
    # X_train shape: (200, 1)
    x_train = X_train.flatten()
    y_train = y_train.flatten()
    
    # Standardize x for better conditioning of the kernel matrix
    x_mean = np.mean(x_train)
    x_std = np.std(x_train)
    if x_std < 1e-8:
        x_std = 1.0
    x_train_norm = (x_train - x_mean) / x_std

    # Precompute distances for efficiency during optimization
    # Note: In 1D, distance is just abs difference
    # But cdist is general and fast enough for N=200
    
    def rbf_kernel(X1, X2, length_scale, signal_var):
        # X1, X2 are 1D arrays
        dist_sq = (X1[:, None] - X2[None, :]) ** 2
        return signal_var * np.exp(-0.5 * dist_sq / (length_scale ** 2))

    def log_marginal_likelihood(params):
        log_l = params[0]
        log_s = params[1]
        
        # Bounds for stability
        if log_l < -10 or log_l > 10:
            return -1e10
        if log_s < -10 or log_s > 10:
            return -100.0 # Allow smaller signal var, but not too small to cause singularity
        
        length_scale = np.exp(log_l)
        signal_var = np.exp(log_s)
        
        K = rbf_kernel(x_train_norm, x_train_norm, length_scale, signal_var)
        
        # Add jitter for numerical stability
        jitter = 1e-6
        K += jitter * np.eye(len(x_train_norm))
        
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return -1e10
            
        # alpha = K^-1 y
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        
        # Log likelihood
        # 0.5 * y^T K^-1 y - 0.5 * log(det(K)) - N/2 * log(2pi)
        # log(det(K)) = 2 * sum(log(diag(L)))
        ll = -0.5 * y_train @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * len(y_train) * np.log(2 * np.pi)
        
        return -ll  # Minimize negative log likelihood

    # Initial guesses
    # Length scale around 1.0 in normalized space
    # Signal variance around var(y)
    initial_log_l = 0.0 
    initial_log_s = np.log(max(np.var(y_train), 1e-6))
    
    best_params = [initial_log_l, initial_log_s]
    best_score = -1e10
    gp_model = None
    
    # Try to optimize hyperparameters
    try:
        # Use multiple restarts if time permits
        max_restarts = 5
        for i in range(max_restarts):
            if time.time() > DEADLINE - 10:
                break
                
            if i == 0:
                x0 = [initial_log_l, initial_log_s]
            else:
                # Random restarts
                x0 = [initial_log_l + np.random.randn() * 0.5, 
                      initial_log_s + np.random.randn() * 0.5]
            
            res = minimize(log_marginal_likelihood, x0, method='L-BFGS-B', 
                           options={'maxiter': 50})
            
            if res.success and res.fun < best_score:
                best_score = res.fun
                best_params = res.x
                
    except Exception:
        pass

    # Build the GP predictor if we have valid params
    # Use the best params found
    log_l, log_s = best_params
    length_scale = np.exp(log_l)
    signal_var = np.exp(log_s)
    
    # If optimization failed or gave bad params, fall back to reasonable defaults
    if not np.isfinite(best_score) or best_score > -10:
        length_scale = 1.0
        signal_var = max(np.var(y_train), 1e-6)
        log_l = 0.0
        log_s = np.log(signal_var)

    # Pre-compute Cholesky for training data with best params
    K_train = rbf_kernel(x_train_norm, x_train_norm, length_scale, signal_var)
    K_train += 1e-6 * np.eye(len(x_train_norm))
    
    try:
        L_train = np.linalg.cholesky(K_train)
        # Solve for alpha
        alpha = np.linalg.solve(L_train.T, np.linalg.solve(L_train, y_train))
        gp_valid = True
    except np.linalg.LinAlgError:
        gp_valid = False

    def predict(X):
        # X shape: (N_test, 1)
        x_test = X.flatten()
        
        # Check time limit, if exceeded, use a simple fallback (linear or constant)
        # But we must return finite values.
        
        if not gp_valid:
            # Fallback: Linear regression
            # We can't easily do lstsq here without precomputing, but we can do a simple mean or linear fit
            # Let's do a quick linear fit on the fly if GP failed? No, that's slow.
            # Better fallback: Return mean of y_train
            return np.full(len(x_test), np.mean(y_train))
        
        # Normalize test points
        x_test_norm = (x_test - x_mean) / x_std
        
        # Compute K_test_train
        K_test_train = rbf_kernel(x_test_norm, x_train_norm, length_scale, signal_var)
        
        # Prediction mean
        # f(x*) = K(x*, X_train) @ alpha
        y_pred = K_test_train @ alpha
        
        # Ensure finite
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
        
        return y_pred

    return predict
