"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 20.0 # Per-equation soft deadline
    
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float).flatten()
    
    n_samples, n_features = X.shape
    
    # Handle constant y
    if np.allclose(y, y[0]):
        const_val = y[0]
        return lambda X_test: np.full(X_test.shape[0], const_val)
        
    # Standardization
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X - X_mean) / X_std
    
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std == 0:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    # Define Matern 5/2 Kernel
    def matern_kernel(X1, X2, length_scales, signal_var):
        # X1: (N, D), X2: (M, D)
        # length_scales: (D,)
        # signal_var: scalar
        # r^2 = sum((x_i - x_j)^2 / l_k^2)
        X1_scaled = X1 / length_scales
        X2_scaled = X2 / length_scales
        sq_dists = cdist(X1_scaled, X2_scaled, metric='sqeuclidean')
        r = np.sqrt(np.maximum(sq_dists, 0))
        K = signal_var * (1 + r + (r**2)/3) * np.exp(-r)
        return K

    # Log Marginal Likelihood for GP
    def log_marginal_likelihood(log_params):
        log_sigma_f = log_params[0]
        log_l = log_params[1:1+n_features]
        log_sigma_n = log_params[1+n_features]
        
        sigma_f = np.exp(log_sigma_f)
        l = np.exp(log_l)
        sigma_n = np.exp(log_sigma_n)
        
        K = matern_kernel(X_norm, X_norm, l, sigma_f)
        K[np.diag_indices_from(K)] += sigma_n**2
        
        try:
            L = np.linalg.cholesky(K + 1e-12 * np.eye(n_samples))
        except np.linalg.LinAlgError:
            return -1e10 # Bad likelihood
            
        alpha = np.linalg.solve(np.linalg.cholesky(K + 1e-12 * np.eye(n_samples)), y_norm) # This is inefficient, better to solve L L^T alpha = y
        # Actually, standard way:
        # L L^T = K
        # K alpha = y -> L L^T alpha = y -> L^T alpha = L^-1 y -> alpha = L^-T (L^-1 y)
        try:
            v = np.linalg.solve(L, y_norm)
            alpha = np.linalg.solve(L.T, v)
        except np.linalg.LinAlgError:
            return -1e10
            
        log_det = 2 * np.sum(np.log(np.diag(L)))
        
        # -0.5 * y^T K^-1 y - 0.5 * log(det(K)) - N/2 * log(2*pi)
        mll = -0.5 * np.dot(y_norm, alpha) - 0.5 * log_det - n_samples/2.0 * np.log(2 * np.pi)
        
        if not np.isfinite(mll):
            return -1e10
            
        return -mll # Minimize negative MLL

    best_predict = None
    gp_success = False
    
    # Initial guess for hyperparameters
    # Assume length scales around 1.0 (in normalized space), signal var 1.0, noise 0.1
    x0 = np.zeros(1 + n_features + 1)
    x0[0] = 0.0 # log_sigma_f
    x0[1:1+n_features] = 0.0 # log_l
    x0[1+n_features] = -2.0 # log_sigma_n
    
    try:
        result = minimize(
            log_marginal_likelihood, 
            x0, 
            method='L-BFGS-B', 
            options={'maxiter': 100, 'maxfun': 100}
        )
        
        if result.success or result.fun < 100: # Threshold for 'good enough'
            log_params = result.x
            log_sigma_f = log_params[0]
            log_l = log_params[1:1+n_features]
            log_sigma_n = log_params[1+n_features]
            
            sigma_f = np.exp(log_sigma_f)
            l = np.exp(log_l)
            sigma_n = np.exp(log_sigma_n)
            
            K = matern_kernel(X_norm, X_norm, l, sigma_f)
            K[np.diag_indices_from(K)] += sigma_n**2
            
            try:
                L = np.linalg.cholesky(K + 1e-12 * np.eye(n_samples))
                v = np.linalg.solve(L, y_norm)
                alpha = np.linalg.solve(L.T, v)
                gp_success = True
            except np.linalg.LinAlgError:
                gp_success = False
                
    except Exception:
        gp_success = False
        
    if not gp_success or time.time() > deadline:
        # Fallback: Kernel Ridge Regression with RBF kernel
        # Grid search for gamma (1/l^2) and alpha (regularization)
        best_score = np.inf
        best_gamma = None
        best_alpha = None
        best_w = None
        
        gammas = np.logspace(-2, 2, 5)
        alphas = np.logspace(-2, 4, 5)
        
        K_train = None # Lazy eval
        
        for g in gammas:
            # RBF kernel: exp(-g * d^2)
            sq_dists = cdist(X_norm, X_norm, metric='sqeuclidean')
            K_rbf = np.exp(-g * sq_dists)
            K_reg = K_rbf + alphas[0] * np.eye(n_samples) # Just to check shape, will do inner loop
            
            for a in alphas:
                K_reg = K_rbf + a * np.eye(n_samples)
                try:
                    w = np.linalg.solve(K_reg, y_norm)
                    # Calculate training MSE (normalized)
                    pred = K_rbf @ w
                    mse = np.mean((pred - y_norm)**2)
                    if mse < best_score:
                        best_score = mse
                        best_gamma = g
                        best_alpha = a
                        best_w = w
                except np.linalg.LinAlgError:
                    continue
                    
        if best_w is not None:
            gamma = best_gamma
            alpha = best_alpha
            w = best_w
            def predict_rbf(X_test):
                X_test_norm = (np.asarray(X_test, dtype=float) - X_mean) / X_std
                sq_dists = cdist(X_test_norm, X_norm, metric='sqeuclidean')
                K_test = np.exp(-gamma * sq_dists)
                y_pred_norm = K_test @ w
                return y_pred_norm * y_std + y_mean
            return predict_rbf
        else:
            # Last resort: Linear regression
            X_aug = np.column_stack([np.ones(n_samples), X_norm])
            w_lin, *_ = np.linalg.lstsq(X_aug, y_norm, rcond=None)
            def predict_lin(X_test):
                X_test_norm = (np.asarray(X_test, dtype=float) - X_mean) / X_std
                X_aug_test = np.column_stack([np.ones(X_test_norm.shape[0]), X_test_norm])
                y_pred_norm = X_aug_test @ w_lin
                return y_pred_norm * y_std + y_mean
            return predict_lin

    # GP Prediction
    def predict_gp(X_test):
        X_test_norm = (np.asarray(X_test, dtype=float) - X_mean) / X_std
        K_train_test = matern_kernel(X_norm, X_test_norm, l, sigma_f)
        # y_pred = K(X_test, X_train) * K(X_train, X_train)^-1 * y_train
        # We have alpha = K_train^-1 * y_train
        y_pred_norm = K_train_test.T @ alpha
        return y_pred_norm * y_std + y_mean
        
    return predict_gp

# EVOLVE-BLOCK-END
