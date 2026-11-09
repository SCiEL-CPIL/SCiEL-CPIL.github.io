"""Gaussian Process Regression for symbolic regression task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def _matern_kernel(X1, X2, length_scale, signal_var, nu=2.5):
    """
    Matern kernel with nu=2.5
    K(x, x') = sigma^2 * (1 + sqrt(3)*r/l) * exp(-sqrt(3)*r/l)
    """
    r = cdist(X1, X2)
    sqrt3 = np.sqrt(3.0)
    z = sqrt3 * r / length_scale
    return signal_var * (1 + z) * np.exp(-z)

def _log_marginal_likelihood(params, X, y):
    """
    Compute negative log marginal likelihood for GP.
    params: [log_signal_var, log_length_scale, log_noise_var]
    """
    log_signal_var, log_length_scale, log_noise_var = params
    signal_var = np.exp(log_signal_var)
    length_scale = np.exp(log_length_scale)
    noise_var = np.exp(log_noise_var)
    
    n = X.shape[0]
    K = _matern_kernel(X, X, length_scale, signal_var)
    K += noise_var * np.eye(n)
    
    try:
        # Cholesky decomposition for stability
        L = np.linalg.cholesky(K)
        # alpha = K^-1 y
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        # log det K = 2 * sum(log(diag(L)))
        log_det_K = 2.0 * np.sum(np.log(np.diag(L)))
        
        # NLML = 0.5 * y^T K^-1 y + 0.5 * log(det(K)) + 0.5 * n * log(2*pi)
        nlml = 0.5 * np.dot(y, alpha) + 0.5 * log_det_K + 0.5 * n * np.log(2.0 * np.pi)
        
        # Add penalty if parameters are too extreme
        if not np.isfinite(nlml):
            return 1e10
            
        return nlml
    except np.linalg.LinAlgError:
        return 1e10

def fit(X_train, y_train):
    """
    Fit a Gaussian Process model with Matern kernel.
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    
    # Standardize X
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_std[X_std < 1e-10] = 1.0
    X_train_norm = (X_train - X_mean) / X_std
    
    # Standardize y
    y_mean = y_train.mean()
    y_std = y_train.std()
    if y_std < 1e-10:
        y_std = 1.0
    y_train_norm = (y_train - y_mean) / y_std
    
    # Initial hyperparameters
    # log(signal_var) ~ log(1) = 0
    # log(length_scale) ~ log(1) = 0
    # log(noise_var) ~ log(1e-6) = -13.8 (small noise since data is noiseless, but keep small for stability)
    x0 = [0.0, 0.0, -13.8]
    
    # Optimize hyperparameters
    result = minimize(_log_marginal_likelihood, x0, args=(X_train_norm, y_train_norm), method='L-BFGS-B')
    
    log_signal_var, log_length_scale, log_noise_var = result.x
    signal_var = np.exp(log_signal_var)
    length_scale = np.exp(log_length_scale)
    noise_var = np.exp(log_noise_var)
    
    # Pre-compute Cholesky factor for training data
    K_train = _matern_kernel(X_train_norm, X_train_norm, length_scale, signal_var)
    K_train += noise_var * np.eye(X_train_norm.shape[0])
    L = np.linalg.cholesky(K_train)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train_norm))
    
    def predict(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        X_test_norm = (X - X_mean) / X_std
        
        K_test_train = _matern_kernel(X_test_norm, X_train_norm, length_scale, signal_var)
        
        # Prediction mean
        mu_norm = K_test_train @ alpha
        mu = mu_norm * y_std + y_mean
        
        return mu.flatten()
    
    return predict
# EVOLVE-BLOCK-END
