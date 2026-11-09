import numpy as np
import time
from scipy.spatial.distance import cdist

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 325.0
    
    X = X_train
    y = y_train
    n, d = X.shape
    
    if np.var(y) < 1e-10:
        mean_y = np.mean(y)
        return lambda X_test: np.full(len(X_test), mean_y)

    # Normalize X
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X - X_mean) / X_std
    
    # Split for validation
    indices = np.arange(n)
    np.random.shuffle(indices)
    val_size = 40
    val_idx = indices[:val_size]
    tr_idx = indices[val_size:]
    
    X_tr = X_norm[tr_idx]
    y_tr = y[tr_idx]
    X_val = X_norm[val_idx]
    y_val = y[val_idx]
    
    # Candidate sigmas
    sigmas = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    
    best_score = np.inf
    best_sigma = 1.0
    best_theta = None
    best_X_tr = None
    
    # Linear features
    X_tr_lin = np.column_stack([X_tr, np.ones(len(X_tr))])
    X_val_lin = np.column_stack([X_val, np.ones(len(X_val))])
    
    for sigma in sigmas:
        # RBF Kernel
        D_tr = cdist(X_tr, X_tr, metric='sqeuclidean')
        K_tr = np.exp(-D_tr / (2 * sigma**2))
        
        # Combined Design Matrix: [K, X_lin]
        # Regularization on RBF part? 
        # Let's use Ridge Regression on the combined features.
        # Lambda for RBF part, 0 for linear? 
        # Simpler: Just use Least Squares on the combined matrix. 
        # If N is large, it might be ill-conditioned. 
        # N=160, d_lin <= 4. Total features ~ 164. 
        # Rank is likely full.
        
        Phi_tr = np.hstack([K_tr, X_tr_lin])
        
        try:
            # Use lstsq for stability
            theta, *_ = np.linalg.lstsq(Phi_tr, y_tr, rcond=None)
        except np.linalg.LinAlgError:
            continue
        
        # Predict
        D_val = cdist(X_val, X_tr, metric='sqeuclidean')
        K_val = np.exp(-D_val / (2 * sigma**2))
        Phi_val = np.hstack([K_val, X_val_lin])
        
        y_pred_val = Phi_val @ theta
        mse = np.mean((y_pred_val - y_val)**2)
        
        if mse < best_score:
            best_score = mse
            best_sigma = sigma
            # Store theta? No, we need to refit on all data.
            # But we need to know the structure.
            
    # Refit with best sigma on all data
    sigma = best_sigma
    D_all = cdist(X_norm, X_norm, metric='sqeuclidean')
    K_all = np.exp(-D_all / (2 * sigma**2))
    X_all_lin = np.column_stack([X_norm, np.ones(n)])
    Phi_all = np.hstack([K_all, X_all_lin])
    
    try:
        theta_final, *_ = np.linalg.lstsq(Phi_all, y, rcond=None)
    except np.linalg.LinAlgError:
        # Fallback to simple linear
        theta_final = np.zeros(Phi_all.shape[1])
        theta_final[-(d+1):] = np.linalg.lstsq(X_all_lin, y, rcond=None)[0]

    def predict(X_test):
        X_test_norm = (X_test - X_mean) / X_std
        D_test = cdist(X_test_norm, X_norm, metric='sqeuclidean')
        K_test = np.exp(-D_test / (2 * sigma**2))
        X_test_lin = np.column_stack([X_test_norm, np.ones(len(X_test))])
        Phi_test = np.hstack([K_test, X_test_lin])
        return Phi_test @ theta_final
        
    return predict
