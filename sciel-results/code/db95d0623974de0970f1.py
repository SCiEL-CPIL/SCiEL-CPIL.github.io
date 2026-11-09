"""Symbolic regression via Kernel Ridge Regression with Cross-Validation."""
import time
import numpy as np

def fit(X_train, y_train):
    """
    Fit a regression model using Kernel Ridge Regression.
    Uses Cross-Validation to select the best RBF bandwidth and regularization.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    n, d = X.shape

    # Normalize inputs to standard normal to stabilize RBF kernel
    # This is crucial for RBF as it is scale-sensitive
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    x_std[x_std == 0] = 1.0  # Avoid division by zero
    X_norm = (X - x_mean) / x_std

    def time_left():
        return deadline - time.time()

    def rbf_kernel(X1, X2, gamma):
        """Compute RBF kernel matrix between X1 and X2."""
        # ||x1 - x2||^2 = ||x1||^2 + ||x2||^2 - 2 * x1 . x2
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * (X1 @ X2.T)
        # Clamp to avoid negative values due to floating point errors
        np.maximum(sq_dists, 0, out=sq_dists)
        return np.exp(-gamma * sq_dists)

    def solve_krr(X_tr, y_tr, X_val, gamma, lam):
        """Solve KRR for training set and predict on validation set."""
        K_tr = rbf_kernel(X_tr, X_tr, gamma)
        K_val = rbf_kernel(X_val, X_tr, gamma)
        
        # Add regularization
        K_reg = K_tr + lam * np.eye(len(X_tr))
        
        try:
            # Solve (K + lam*I) w = y
            w = np.linalg.solve(K_reg, y_tr)
            pred = K_val @ w
            return np.mean((pred - y_val)**2)
        except np.linalg.LinAlgError:
            return np.inf

    # Define search grid
    # Gamma (1 / sigma^2). Small gamma = wide smooth kernel, Large gamma = narrow wiggly kernel.
    # Log spaced from 1e-4 to 1e4
    gamma_grid = np.logspace(-4, 4, 50)
    # Lambda (regularization). Log spaced from 1e-6 to 1e6
    lam_grid = np.logspace(-6, 6, 20)
    
    best_cv_mse = np.inf
    best_gamma = None
    best_lam = None
    
    # Cross-Validation Setup
    # Use 5-fold CV
    n_folds = 5
    indices = np.arange(n)
    np.random.seed(42) # Fixed seed for reproducibility of fold splits
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    
    # Search over grid
    for gamma in gamma_grid:
        if time_left() < 5.0:
            break
            
        # Pre-compute full kernel matrix for this gamma? 
        # No, because CV splits change the training set.
        # However, we can pre-compute the full N x N kernel matrix K_full.
        # Then for each fold, we extract submatrices.
        K_full = rbf_kernel(X_norm, X_norm, gamma)
        
        for lam in lam_grid:
            cv_errors = []
            for i in range(n_folds):
                val_idx = folds[i]
                tr_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
                
                X_tr = X_norm[tr_idx]
                y_tr = y[tr_idx]
                X_val = X_norm[val_idx]
                y_val = y[val_idx]
                
                # Extract submatrices
                K_tr = K_full[np.ix_(tr_idx, tr_idx)]
                K_val = K_full[np.ix_(val_idx, tr_idx)]
                
                K_reg = K_tr + lam * np.eye(len(tr_idx))
                
                try:
                    w = np.linalg.solve(K_reg, y_tr)
                    pred = K_val @ w
                    cv_errors.append(np.mean((pred - y_val)**2))
                except np.linalg.LinAlgError:
                    cv_errors.append(np.inf)
            
            mean_cv_mse = np.mean(cv_errors)
            
            if mean_cv_mse < best_cv_mse:
                best_cv_mse = mean_cv_mse
                best_gamma = gamma
                best_lam = lam
                
    # If no parameters found (should not happen), fallback
    if best_gamma is None:
        best_gamma = 1.0
        best_lam = 1.0

    # Final Fit on all data with best parameters
    K_train = rbf_kernel(X_norm, X_norm, best_gamma)
    K_reg_final = K_train + best_lam * np.eye(n)
    
    try:
        w_final = np.linalg.solve(K_reg_final, y)
    except np.linalg.LinAlgError:
        # Fallback to linear least squares if KRR fails
        w_final = np.linalg.lstsq(K_train, y, rcond=None)[0]

    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        # Normalize test inputs using training statistics
        X_test_norm = (X_test - x_mean) / x_std
        
        # Compute kernel between test and training points
        K_test = rbf_kernel(X_test_norm, X_norm, best_gamma)
        
        try:
            pred = K_test @ w_final
            # Ensure finite outputs
            pred = np.where(np.isfinite(pred), pred, 0.0)
            return pred
        except Exception:
            return np.zeros(len(X_test))

    return predict
