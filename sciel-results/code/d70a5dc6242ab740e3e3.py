"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
import time

def fit(X_train, y_train):
    start_time = time.time()
    # Reserve 50s for overhead, use 70s per equation (4*70=280s total)
    deadline = start_time + 70.0 

    X = np.asarray(X_train).flatten()
    y = np.asarray(y_train).flatten()
    
    # Normalize X to [0, 1] to stabilize kernel scaling
    x_min = X.min()
    x_max = X.max()
    range_x = x_max - x_min
    if range_x > 1e-10:
        X_norm = (X - x_min) / range_x
    else:
        X_norm = np.zeros_like(X)
    
    # Center y to help kernel fit deviations
    y_mean = np.mean(y)
    y_centered = y - y_mean
    
    # Matern 5/2 Kernel
    def matern52_kernel(X1, X2, length_scale, signal_var):
        # X1, X2 are 1D arrays
        # Compute distance matrix
        dist = np.sqrt(np.sum((X1[:, np.newaxis] - X2[np.newaxis, :])**2, axis=1))
        r = dist / length_scale
        return signal_var * (1 + np.sqrt(5)*r + (5/3)*r**2) * np.exp(-np.sqrt(5)*r)
    
    def log_marginal_likelihood(params):
        if time.time() > deadline:
            return 1e10 # Return high cost to stop
        
        length_scale, log_signal_var = params
        signal_var = np.exp(log_signal_var)
        
        K = matern52_kernel(X_norm, X_norm, length_scale, signal_var)
        # Add small noise for numerical stability
        K += 1e-8 * np.eye(len(X_norm))
        
        try:
            L = np.linalg.cholesky(K)
            # Solve K alpha = y
            alpha = cho_solve((L, True), y_centered)
            # Calculate negative log marginal likelihood
            lml = -0.5 * y_centered @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * len(y_centered) * np.log(2*np.pi)
            return -lml
        except np.linalg.LinAlgError:
            return 1e10
    
    best_params = None
    best_score = np.inf
    
    # Random restarts for hyperparameter optimization
    n_restarts = 15
    for i in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Initialize length scale in [0.01, 2.0] and signal var in [0.1, 10.0]
        l0 = np.log(np.random.uniform(0.01, 2.0))
        sv0 = np.log(np.random.uniform(0.1, 10.0))
        
        # Use L-BFGS-B
        res = minimize(log_marginal_likelihood, [l0, sv0], method='L-BFGS-B', 
                       bounds=[(0.01, 2.0), (-2, 2)], options={'maxiter': 50, 'ftol': 1e-6})
        
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
            
    # If optimization failed, use reasonable defaults
    if best_params is None:
        length_scale = 0.5
        signal_var = 1.0
    else:
        length_scale = best_params[0]
        signal_var = np.exp(best_params[1])
        
    # Precompute for prediction
    K_train = matern52_kernel(X_norm, X_norm, length_scale, signal_var)
    K_train += 1e-8 * np.eye(len(X_norm))
    
    L = np.linalg.cholesky(K_train)
    alpha = cho_solve((L, True), y_centered)
    
    def predict(X_test):
        X_test = np.asarray(X_test).flatten()
        if X_test.size == 0:
            return np.array([])
        
        # Normalize test X
        if range_x > 1e-10:
            X_test_norm = (X_test - x_min) / range_x
        else:
            X_test_norm = np.zeros_like(X_test)
            
        K_test_train = matern52_kernel(X_test_norm, X_norm, length_scale, signal_var)
        
        y_pred_centered = K_test_train @ alpha
        y_pred = y_pred_centered + y_mean
        
        return y_pred
        
    return predict
# EVOLVE-BLOCK-END
