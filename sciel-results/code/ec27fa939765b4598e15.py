import numpy as np
from scipy.linalg import solve

def fit(X_train, y_train):
    # Normalize inputs to [0, 1] range to stabilize kernel parameters
    min_vals = X_train.min(axis=0)
    max_vals = X_train.max(axis=0)
    ranges = max_vals - min_vals
    # Avoid division by zero
    ranges[ranges == 0] = 1.0
    X_norm = (X_train - min_vals) / ranges
    
    # Kernel parameters to search
    # RBF gamma: controls width. For normalized [0,1] data, 1.0 to 10.0 is typical.
    # Alpha: regularization. Small values for low bias.
    gammas = [1.0, 5.0, 10.0, 20.0]
    alphas = [1e-4, 1e-3, 1e-2, 1e-1]
    
    best_pred = None
    best_mse = np.inf
    best_params = None
    
    # We can use a small validation set to select params, or just pick a robust one.
    # Since data is noiseless, we want to avoid overfitting to the specific noise-free points 
    # if the model is too flexible, but usually low alpha is good.
    # Let's do a simple split for selection if time permits, but 200 points is small.
    # Let's just try a few and pick the one with lowest training error? 
    # No, that will always be the most overfit one.
    # Better: Use a heuristic. A moderate gamma and small alpha.
    # Or, since it's noiseless, we can just use a very small alpha (1e-6) 
    # and a gamma that covers the range.
    
    # Let's try a fixed, robust configuration first to ensure speed and stability.
    # K = K_poly + K_rbf
    # K_poly(x, y) = (x.dot(y) + 1)^3
    # K_rbf(x, y) = exp(-gamma * ||x-y||^2)
    
    def compute_kernel(X1, X2, gamma):
        # Polynomial part (degree 3)
        K_poly = (X1 @ X2.T + 1.0) ** 3
        # RBF part
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * (X1 @ X2.T)
        sq_dists = np.maximum(sq_dists, 0) # Numerical stability
        K_rbf = np.exp(-gamma * sq_dists)
        return K_poly + K_rbf

    # We will pick parameters based on a quick 5-fold CV on the training set
    # to avoid overfitting, even though noise is 0, the test set is separate.
    # With 200 points, CV is cheap.
    
    indices = np.arange(len(X_train))
    best_cv_mse = np.inf
    best_gamma = 5.0
    best_alpha = 1e-3
    
    # To save time, we can reduce the grid size.
    # 4 gammas * 4 alphas = 16 configs.
    # 5 folds * 16 configs = 80 solves.
    # 80 * (160^3) ~ 80 * 4e6 = 3.2e8 ops. Fast.
    
    for gamma in gammas:
        for alpha in alphas:
            mse_sum = 0
            for fold in range(5):
                val_idx = indices[(indices % 5) == fold]
                train_idx = indices[(indices % 5) != fold]
                
                X_tr = X_norm[train_idx]
                y_tr = y_train[train_idx]
                X_val = X_norm[val_idx]
                y_val = y_train[val_idx]
                
                K_tr = compute_kernel(X_tr, X_tr, gamma)
                K_val = compute_kernel(X_val, X_tr, gamma)
                
                # Solve (K + alpha I) w = y
                try:
                    w = solve(K_tr + alpha * np.eye(len(X_tr)), y_tr, assume_a='pos')
                    y_pred = K_val @ w
                    mse = np.mean((y_pred - y_val)**2)
                    mse_sum += mse
                except:
                    mse_sum = np.inf
                    break
            
            avg_mse = mse_sum / 5
            if avg_mse < best_cv_mse:
                best_cv_mse = avg_mse
                best_gamma = gamma
                best_alpha = alpha
                
    # Final fit with best params on all data
    K_train = compute_kernel(X_norm, X_norm, best_gamma)
    w = solve(K_train + best_alpha * np.eye(len(X_train)), y_train, assume_a='pos')
    
    def predict(X):
        # Normalize X using training stats
        X_norm_test = (X - min_vals) / ranges
        K_test = compute_kernel(X_norm_test, X_norm, best_gamma)
        return K_test @ w
        
    return predict
