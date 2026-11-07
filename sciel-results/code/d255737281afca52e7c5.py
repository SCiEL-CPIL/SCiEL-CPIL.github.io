"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm

def fit(X_train, y_train):
    start_time = time.time()
    DEADLINE = 330.0 - 2.0  # Safety margin
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    # Normalize inputs to [0, 1] to help kernel optimization
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0
    X_norm = (X - X_min) / X_range
    
    # Normalize target for numerical stability in GP
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std == 0:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std

    n = X_norm.shape[0]
    d = X_norm.shape[1]

    # Precompute pairwise squared distances for Matern kernel
    # D2[i, j] = ||x_i - x_j||^2
    diff = X_norm[:, np.newaxis, :] - X_norm[np.newaxis, :, :]
    D2 = np.sum(diff * diff, axis=-1)

    def matern52_kernel(D2, length_scale, variance):
        r = np.sqrt(D2)
        return variance * (1.0 + np.sqrt(3.0) * r / length_scale) * np.exp(-np.sqrt(3.0) * r / length_scale)

    def neg_log_likelihood(params):
        # params: [log_length_scale, log_variance, log_noise]
        ls = np.exp(params[0])
        var = np.exp(params[1])
        noise = np.exp(params[2])
        
        K = matern52_kernel(D2, ls, var)
        K += noise * np.eye(n)
        
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            ll = -0.5 * (y_norm @ alpha + log_det + n * np.log(2 * np.pi))
            return -ll
        except np.linalg.LinAlgError:
            return 1e10

    # Initial guess: length_scale=1.0, variance=1.0, noise=1e-6
    # Since inputs are normalized to [0,1], ls=1 is reasonable
    x0 = np.array([0.0, 0.0, -10.0])
    
    best_params = x0
    best_ll = neg_log_likelihood(x0)
    
    # Optimization
    if time.time() < DEADLINE:
        try:
            res = minimize(neg_log_likelihood, x0, method='L-BFGS-B', options={'maxiter': 50, 'maxfun': 50})
            if res.fun < best_ll:
                best_params = res.x
                best_ll = res.fun
        except Exception:
            pass

    # Refine if time permits (optional, but usually one good run is enough for 200 pts)
    # We can try a few restarts if time allows, but L-BFGS on 3 params is fast.
    # Let's just use the best found.
    
    ls = np.exp(best_params[0])
    var = np.exp(best_params[1])
    noise = np.exp(best_params[2])
    
    K = matern52_kernel(D2, ls, var)
    K += noise * np.eye(n)
    
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
    except np.linalg.LinAlgError:
        # Fallback to polynomial if GP fails
        def features(X):
            cols = [np.ones(len(X))] + [X[:, j] for j in range(X.shape[1])] + [X[:, i] * X[:, j] for i in range(X.shape[1]) for j in range(i, X.shape[1])]
            return np.column_stack(cols)
        w, *_ = np.linalg.lstsq(features(X), y, rcond=None)
        return lambda X: features(X) @ w

    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        X_test_norm = (X_test - X_min) / X_range
        
        # Kernel between test and train
        diff_test = X_test_norm[:, np.newaxis, :] - X_norm[np.newaxis, :, :]
        D2_test = np.sum(diff_test * diff_test, axis=-1)
        
        K_star = matern52_kernel(D2_test, ls, var)
        
        # Mean prediction
        mean_norm = K_star @ alpha
        mean = mean_norm * y_std + y_mean
        
        return mean

    return predict
# EVOLVE-BLOCK-END
