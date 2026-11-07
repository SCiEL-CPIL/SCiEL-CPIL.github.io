import numpy as np
from scipy.spatial.distance import cdist

def fit(X_train, y_train):
    X = np.asarray(X_train, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y_train, dtype=float)
    
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
    
    n_samples, n_features = X.shape
    
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X - X_mean) / X_std
    
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std == 0:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    D = cdist(X_norm, X_norm, metric='euclidean')
    D_sq = D ** 2
    
    triu_indices = np.triu_indices(n_samples, k=1)
    dists_sq = D_sq[triu_indices]
    if len(dists_sq) > 0:
        median_dist_sq = np.median(dists_sq)
    else:
        median_dist_sq = 1.0
    
    gamma_heuristic = 1.0 / (2.0 * median_dist_sq)
    
    gamma_grid = np.logspace(np.log10(max(gamma_heuristic, 1e-4)) - 2,
                              np.log10(min(gamma_heuristic * 100, 1e4)) + 2, 15)
    gamma_grid = np.unique(np.concatenate([gamma_grid, [0.01, 0.1, 1.0, 10.0, 100.0]]))
    
    alpha_grid = np.logspace(-2, 3, 12)
    
    n_folds = 5
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    
    best_mse = np.inf
    best_gamma = gamma_heuristic
    best_alpha = 1.0
    
    for gamma in gamma_grid:
        K_full = np.exp(-gamma * D_sq)
        
        for alpha in alpha_grid:
            cv_mse_sum = 0.0
            valid = True
            for i in range(n_folds):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
                
                K_val_train = K_full[np.ix_(val_idx, train_idx)]
                K_train_train = K_full[np.ix_(train_idx, train_idx)]
                y_val = y_norm[val_idx]
                y_train = y_norm[train_idx]
                
                try:
                    A = K_train_train + alpha * np.eye(len(train_idx))
                    beta = np.linalg.solve(A, y_train)
                    y_pred_val = K_val_train @ beta
                    mse = np.mean((y_val - y_pred_val)**2)
                    cv_mse_sum += mse
                except np.linalg.LinAlgError:
                    valid = False
                    break
            
            if not valid:
                continue
                
            mean_cv_mse = cv_mse_sum / n_folds
            if mean_cv_mse < best_mse:
                best_mse = mean_cv_mse
                best_gamma = gamma
                best_alpha = alpha
    
    K_full_best = np.exp(-best_gamma * D_sq)
    try:
        A = K_full_best + best_alpha * np.eye(n_samples)
        beta = np.linalg.solve(A, y_norm)
    except np.linalg.LinAlgError:
        beta, *_ = np.linalg.lstsq(K_full_best + best_alpha * np.eye(n_samples), y_norm, rcond=None)
    
    def predict(X_test):
        X_test = np.asarray(X_test, dtype=float)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
        X_test_norm = (X_test - X_mean) / X_std
        
        D_test = cdist(X_test_norm, X_norm, metric='euclidean')
        D_test_sq = D_test ** 2
        K_test_train = np.exp(-best_gamma * D_test_sq)
        
        y_pred_norm = K_test_train @ beta
        y_pred = y_pred_norm * y_std + y_mean
        
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return y_pred

    return predict
