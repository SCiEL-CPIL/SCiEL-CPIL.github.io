import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 70 # 70s per equation to be safe (4 * 70 = 280 < 300)
    
    X = X_train.flatten()
    y = y_train.flatten()
    
    # Normalize y to help optimization
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std < 1e-12:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    # Kernel function: RBF
    def kernel(X1, X2, ell, sf):
        # X1, X2 are 1D arrays
        diff = X1[:, None] - X2[None, :]
        return sf**2 * np.exp(-0.5 * diff**2 / ell**2)
    
    # Negative Log Marginal Likelihood
    def nlml(params):
        if time.time() > deadline:
            return 1e10
        ell, log_sf, log_noise = params
        ell = np.exp(ell)
        sf = np.exp(log_sf)
        noise = np.exp(log_noise)
        
        K = kernel(X, X, ell, sf)
        K[np.diag_indices_from(K)] += noise
        
        try:
            L, _ = cho_factor(K, lower=True, check_finite=False)
            alpha = cho_solve((L, True), y_norm, check_finite=False)
            # NLML = 0.5 * y^T K^-1 y + 0.5 * log det K + 0.5 * N * log(2*pi)
            # log det K = 2 * sum(log(diag(L)))
            log_det_K = 2.0 * np.sum(np.log(np.diag(L)))
            nlml_val = 0.5 * np.dot(y_norm, alpha) + 0.5 * log_det_K + 0.5 * len(y_norm) * np.log(2.0 * np.pi)
            return nlml_val
        except:
            return 1e10

    # Initial guess
    # Length scale: estimate from median distance
    dists = np.abs(X[:, None] - X[None, :])
    np.fill_diagonal(dists, np.inf)
    med_dist = np.median(dists)
    if med_dist == 0: med_dist = 1.0
    
    init_ell = np.log(med_dist)
    init_sf = np.log(1.0)
    init_noise = np.log(1e-6) # Very small noise for noiseless data
    
    x0 = [init_ell, init_sf, init_noise]
    
    try:
        res = minimize(nlml, x0, method='L-BFGS-B', 
                       bounds=[(np.log(1e-3), np.log(100)), (-10, 10), (-20, 0)],
                       options={'maxiter': 50, 'ftol': 1e-6})
        
        if res.success or res.fun < 1e9:
            best_params = res.x
        else:
            # Fallback to initial or simple KRR
            best_params = x0
            
    except Exception:
        best_params = x0

    if time.time() > deadline - 5:
        # Fallback to simple polynomial if we are running out of time
        # Or just use the last computed model if available? 
        # Let's just return a constant predictor if we are in a hurry
        return lambda X: np.full((X.shape[0], 1), y_mean)

    ell = np.exp(best_params[0])
    sf = np.exp(best_params[1])
    noise = np.exp(best_params[2])
    
    K = kernel(X, X, ell, sf)
    K[np.diag_indices_from(K)] += noise
    
    try:
        L, _ = cho_factor(K, lower=True, check_finite=False)
        alpha = cho_solve((L, True), y_norm, check_finite=False)
    except:
        # Fallback to KRR with ridge
        K[np.diag_indices_from(K)] += 1e-6
        alpha = np.linalg.solve(K, y_norm)
        
    def predict(X_test):
        X_test = X_test.flatten()
        K_s = kernel(X_test, X, ell, sf)
        # Predict y_norm
        y_pred_norm = K_s @ alpha
        y_pred = y_pred_norm * y_std + y_mean
        return y_pred.reshape(-1, 1)
        
    return predict
