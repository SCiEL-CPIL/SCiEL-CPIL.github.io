import numpy as np
from scipy.spatial.distance import cdist

def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    
    n, d = X_train.shape
    
    # Standardize X
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_std[X_std < 1e-10] = 1.0
    X_scaled = (X_train - X_mean) / X_std
    
    # Center Y
    y_mean = y_train.mean()
    y_centered = y_train - y_mean
    
    # Grid search
    gammas = [0.01, 0.1, 1.0, 10.0, 100.0]
    alphas = [1e-6, 1e-4, 1e-2, 1e-1, 1.0]
    
    best_gamma = None
    best_alpha = None
    best_mse = np.inf
    best_weights = None
    
    for gamma in gammas:
        dists_sq = cdist(X_scaled, X_scaled, metric='sqeuclidean')
        K = np.exp(-gamma * dists_sq)
        
        for alpha in alphas:
            K_reg = K + alpha * np.eye(n)
            
            try:
                w = np.linalg.solve(K_reg, y_centered)
                pred_centered = K @ w
                mse = np.mean((pred_centered - y_centered) ** 2)
                
                # We want low MSE. Since data is noiseless, MSE -> 0 is good.
                # But we should avoid numerical instability.
                if mse < best_mse:
                    best_mse = mse
                    best_gamma = gamma
                    best_alpha = alpha
                    best_weights = w
            except np.linalg.LinAlgError:
                continue

    if best_gamma is None:
        # Fallback
        def predict(X):
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            return np.full(X.shape[0], y_mean)
        return predict

    gamma = best_gamma
    weights = best_weights
    X_train_scaled = X_scaled
    
    def predict(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        X_scaled_test = (X - X_mean) / X_std
        dists_sq_test_train = cdist(X_scaled_test, X_train_scaled, metric='sqeuclidean')
        K_test = np.exp(-gamma * dists_sq_test_train)
        
        preds_centered = K_test @ weights
        preds = preds_centered + y_mean
        
        return preds.flatten()

    return predict
