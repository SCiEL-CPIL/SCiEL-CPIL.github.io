# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

# Global time tracker
_total_start = None
_time_limit = 300.0  # Safety margin before 330s kill

def _elapsed():
    global _total_start
    if _total_start is None:
        _total_start = time.time()
    return time.time() - _total_start

def _remaining():
    return _time_limit - _elapsed()

def fit(X_train, y_train):
    global _total_start
    if _total_start is None:
        _total_start = time.time()
    
    X = X_train.astype(np.float64)
    y = y_train.astype(np.float64)
    n, d = X.shape
    
    # Normalize X and y for numerical stability in kernel computation
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-10] = 1.0
    X_norm = (X - X_mean) / X_std
    
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-10:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std

    # Precompute pairwise distances for RBF kernel
    # K_rbf(X, X) = exp(-gamma * ||xi - xj||^2)
    # We can compute squared distances efficiently
    
    def compute_kernel(X_a, X_b, gamma, poly_degree=3, poly_coeff=1.0):
        # RBF Kernel
        dists = cdist(X_a, X_b)
        K_rbf = np.exp(-gamma * dists**2)
        
        # Poly Kernel: (gamma * X_a @ X_b.T + coef0) ** degree
        K_poly = (poly_coeff * (X_a @ X_b.T) + 1.0) ** poly_degree
        
        # Combine: K = K_rbf + K_poly + I (for regularization)
        # Actually, let's stick to a single RBF kernel with adaptive gamma first, 
        # as it is more robust for arbitrary functions. 
        # Adding poly can help for polynomial-like functions.
        # Let's try a mix: K = w1*K_rbf + w2*K_poly
        # To keep it simple and fast, we'll just use RBF and search gamma.
        # If RBF fails, we can fallback.
        return K_rbf

    # Strategy: Kernel Ridge Regression with Hyperparameter Search
    # Objective: Minimize MSE on a validation set
    # Parameters: gamma (RBF width), lambda (Regularization)
    
    # Split data into train/val for hyperparameter selection
    # Use 80% train, 20% val
    np.random.seed(42)
    indices = np.random.permutation(n)
    split = int(0.8 * n)
    idx_train = indices[:split]
    idx_val = indices[split:]
    
    X_tr, y_tr = X_norm[idx_train], y_norm[idx_train]
    X_va, y_va = X_norm[idx_val], y_norm[idx_val]
    
    # Grid search for gamma and lambda
    # gamma range: small (wide kernel) to large (narrow kernel)
    # Since X is normalized, typical gamma is 1/d to 1
    gamma_candidates = np.logspace(-2, 2, 15)
    lambda_candidates = np.logspace(-2, 4, 15)
    
    best_val_mse = np.inf
    best_gamma = 1.0
    best_lambda = 1.0
    
    # Compute full kernel matrix for training set (for efficiency, we reuse)
    # Note: We need K_tr_tr, K_tr_va
    # K_tr_tr = K(X_tr, X_tr)
    # K_tr_va = K(X_tr, X_va)
    
    # To save time, we compute these inside the loop only if necessary, 
    # or precompute for each gamma.
    # cdist is O(N^2 * D). N=160, D<=3. This is fast.
    
    for gamma in gamma_candidates:
        if _remaining() < 5.0:
            break
            
        K_tr_tr = np.exp(-gamma * cdist(X_tr, X_tr)**2)
        K_tr_va = np.exp(-gamma * cdist(X_tr, X_va)**2)
        
        for lam in lambda_candidates:
            # Solve (K + lam*I) @ w = y_tr
            A = K_tr_tr + lam * np.eye(len(X_tr))
            try:
                w = np.linalg.solve(A, y_tr)
            except np.linalg.LinAlgError:
                w = np.linalg.lstsq(A, y_tr, rcond=None)[0]
            
            # Predict on validation set
            y_va_pred = K_tr_va @ w
            
            # Calculate MSE
            mse = np.mean((y_va_pred - y_va)**2)
            
            if mse < best_val_mse:
                best_val_mse = mse
                best_gamma = gamma
                best_lambda = lam

    # Refit on all data with best hyperparameters
    K_all = np.exp(-best_gamma * cdist(X_norm, X_norm)**2)
    A = K_all + best_lambda * np.eye(n)
    try:
        w_all = np.linalg.solve(A, y_norm)
    except np.linalg.LinAlgError:
        w_all = np.linalg.lstsq(A, y_norm, rcond=None)[0]

    def predict(X_test):
        X_t = X_test.astype(np.float64)
        X_t_norm = (X_t - X_mean) / X_std
        
        # Compute kernel between test points and training points
        K_test = np.exp(-best_gamma * cdist(X_t_norm, X_norm)**2)
        
        y_pred_norm = K_test @ w_all
        y_pred = y_pred_norm * y_std + y_mean
        
        # Safety: ensure finite
        mask = ~np.isfinite(y_pred)
        if np.any(mask):
            y_pred = y_pred.copy()
            y_pred[mask] = y_mean
            
        return y_pred

    return predict
# EVOLVE-BLOCK-END
