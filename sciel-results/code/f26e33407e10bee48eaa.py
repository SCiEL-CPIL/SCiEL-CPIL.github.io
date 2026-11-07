import numpy as np
from scipy.spatial.distance import cdist

def fit(X_train, y_train):
    X = X_train
    y = y_train
    
    # Normalize X for better kernel conditioning
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X - X_mean) / X_std

    def rbf_kernel(X1, X2, gamma):
        # X1: (n, d), X2: (m, d)
        # dist^2 = sum(x1^2) + sum(x2^2) - 2 * x1 @ x2.T
        sq_norms1 = np.sum(X1**2, axis=1)
        sq_norms2 = np.sum(X2**2, axis=1)
        dist_sq = sq_norms1[:, None] + sq_norms2[None, :] - 2 * X1 @ X2.T
        dist_sq = np.maximum(dist_sq, 0)
        return np.exp(-gamma * dist_sq)

    # Hyperparameter grid
    gammas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    alphas = [1e-6, 1e-4, 1e-2, 1.0, 10.0]
    
    # Cross-validation to select best gamma and alpha
    n = len(X_norm)
    k_folds = 5
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k_folds)
    
    best_score = np.inf
    best_gamma = 1.0
    best_alpha = 1e-2
    
    for gamma in gammas:
        for alpha in alphas:
            cv_errors = []
            for i in range(k_folds):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != i])
                
                X_tr = X_norm[train_idx]
                y_tr = y[train_idx]
                X_val = X_norm[val_idx]
                y_val = y[val_idx]
                
                K_train = rbf_kernel(X_tr, X_tr, gamma)
                K_val = rbf_kernel(X_val, X_tr, gamma)
                
                # Solve (K + alpha*I) w = y
                try:
                    w = np.linalg.solve(K_train + alpha * np.eye(len(X_tr)), y_tr)
                    pred_val = K_val @ w
                    mse = np.mean((pred_val - y_val)**2)
                    cv_errors.append(mse)
                except np.linalg.LinAlgError:
                    cv_errors.append(np.inf)
            
            mean_cv_error = np.mean(cv_errors)
            if mean_cv_error < best_score:
                best_score = mean_cv_error
                best_gamma = gamma
                best_alpha = alpha

    # Final fit on all data
    K_train = rbf_kernel(X_norm, X_norm, best_gamma)
    try:
        w = np.linalg.solve(K_train + best_alpha * np.eye(len(X_norm)), y)
    except np.linalg.LinAlgError:
        # Fallback to least squares if singular
        w, *_ = np.linalg.lstsq(K_train + best_alpha * np.eye(len(X_norm)), y, rcond=None)

    def predict(X):
        X_test = (X - X_mean) / X_std
        K_test = rbf_kernel(X_test, X_norm, best_gamma)
        return K_test @ w

    return predict
