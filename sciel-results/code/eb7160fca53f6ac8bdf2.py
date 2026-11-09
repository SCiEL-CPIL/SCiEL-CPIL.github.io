"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 250  # Safety margin for 300s limit
    
    # Flatten X for 1D processing
    X = X_train.reshape(-1, 1)
    y = y_train.reshape(-1)
    n = len(y)
    
    # Kernel function: Matérn 5/2
    def matern52_kernel(X1, X2, length_scale, variance):
        d = cdist(X1, X2, 'euclidean')
        sqrt3d = np.sqrt(3) * d / length_scale
        K = variance * (1 + sqrt3d + (sqrt3d ** 2) / 3) * np.exp(-sqrt3d)
        return K

    # Log marginal likelihood
    def log_marginal_likelihood(params):
        if time.time() > deadline:
            return 0.0
        length_scale, variance, noise = params
        # Constrain parameters to be positive
        if length_scale <= 0 or variance <= 0 or noise <= 0:
            return 0.0
        
        K = matern52_kernel(X, X, length_scale, variance)
        K += noise * np.eye(n)
        
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            log_det = 2 * np.sum(np.log(np.diag(L)))
            lml = -0.5 * y.dot(alpha) - 0.5 * log_det - 0.5 * n * np.log(2 * np.pi)
            return -lml # Minimize negative log marginal likelihood
        except np.linalg.LinAlgError:
            return 1e10

    # Initial guess
    # Estimate length scale from data range
    x_range = np.max(X) - np.min(X)
    initial_ls = x_range / 5.0 if x_range > 0 else 1.0
    initial_var = np.var(y) + 1e-6
    initial_noise = 1e-6
    
    x0 = np.array([initial_ls, initial_var, initial_noise])
    
    # Optimize hyperparameters
    best_params = x0
    best_lml = 1e10
    try:
        res = minimize(log_marginal_likelihood, x0, method='Nelder-Mead', 
                       options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 200})
        if time.time() < deadline:
            if res.fun < best_lml:
                best_params = res.x
                best_lml = res.fun
    except Exception:
        pass

    # If optimization failed or didn't improve, use initial guess or fallback
    if best_lml == 1e10:
        best_params = x0

    length_scale, variance, noise = best_params
    
    # Precompute for prediction
    K_train = matern52_kernel(X, X, length_scale, variance)
    K_train += noise * np.eye(n)
    
    try:
        L = np.linalg.cholesky(K_train)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    except np.linalg.LinAlgError:
        # Fallback to polynomial if GP fails
        def features(X):
            cols = [np.ones(len(X))] + [X[:, j] for j in range(X.shape[1])] + [X[:, i] * X[:, j] for i in range(X.shape[1]) for j in range(i, X.shape[1])]
            return np.column_stack(cols)
        w, *_ = np.linalg.lstsq(features(X_train), y_train, rcond=None)
        return lambda X: features(X) @ w

    def predict(X_pred):
        X_pred = X_pred.reshape(-1, 1)
        K_star = matern52_kernel(X_pred, X, length_scale, variance)
        # Mean prediction
        mu = K_star.dot(alpha)
        return mu

    return predict
# EVOLVE-BLOCK-END
