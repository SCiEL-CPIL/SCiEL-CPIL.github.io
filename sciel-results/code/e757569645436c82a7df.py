import numpy as np

def fit(X_train, y_train):
    X = X_train
    y = y_train
    
    if X.shape[1] > 1:
        X = X[:, 0]
        
    # Normalize X for stable hyperparameter search
    x_mean = np.mean(X)
    x_std = np.std(X)
    if x_std < 1e-8:
        x_std = 1.0
    X_norm = (X - x_mean) / x_std
    
    # GPR with RBF Kernel
    # Gamma grid in normalized space
    # Standard practice: gamma in [0.01, 100]
    gammas = np.logspace(-2, 2, 20)
    
    best_mse = np.inf
    best_alpha = None
    best_gamma = 1.0
    sigma = 1e-6
    
    for gamma in gammas:
        diff = X_norm[:, np.newaxis] - X_norm[np.newaxis, :]
        sq_dist = diff ** 2
        K = np.exp(-gamma * sq_dist)
        K_reg = K + sigma * np.eye(len(X_norm))
        
        try:
            alpha = np.linalg.solve(K_reg, y)
        except np.linalg.LinAlgError:
            alpha = np.linalg.lstsq(K_reg, y, rcond=None)[0]
            
        y_pred = K @ alpha
        mse = np.mean((y_pred - y) ** 2)
        
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
            best_gamma = gamma

    def predict(X_test):
        X_t = X_test
        if X_t.shape[1] > 1:
            X_t = X_t[:, 0]
            
        X_t_norm = (X_t - x_mean) / x_std
        
        diff_t = X_t_norm[:, np.newaxis] - X_norm[np.newaxis, :]
        sq_dist_t = diff_t ** 2
        K_test = np.exp(-best_gamma * sq_dist_t)
        
        y_pred = K_test @ best_alpha
        return y_pred

    return predict
