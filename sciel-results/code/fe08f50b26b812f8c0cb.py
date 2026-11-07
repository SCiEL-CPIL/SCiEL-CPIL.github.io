"""Symbolic regression via Kernel Ridge Regression with LOOCV selection."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = None
_TIME_LIMIT = 330.0
_SAFETY = 10.0


def _time_left():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _TIME_LIMIT - _SAFETY
    return _DEADLINE - time.time()


def _rbf_kernel(X, Y, sigma):
    """Compute RBF kernel matrix between X and Y."""
    # X: (n, d), Y: (m, d)
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y
    xx = np.sum(X * X, axis=1)
    yy = np.sum(Y * Y, axis=1)
    d2 = xx[:, None] + yy[None, :] - 2.0 * (X @ Y.T)
    # Clip negative values due to numerical errors
    d2 = np.maximum(d2, 0.0)
    return np.exp(-d2 / (2.0 * sigma * sigma))


def _loocv_rbf(x_train, y_train, sigma, lam):
    """
    Compute LOOCV MSE for RBF Kernel Ridge Regression.
    x_train: (N, 1)
    y_train: (N,)
    sigma: kernel bandwidth
    lam: regularization strength
    """
    N = x_train.shape[0]
    K = _rbf_kernel(x_train, x_train, sigma)
    A = K + lam * np.eye(N)
    
    try:
        # Solve A alpha = y
        alpha = np.linalg.solve(A, y_train)
    except np.linalg.LinAlgError:
        return np.inf
        
    y_pred = K @ alpha
    
    # Hat matrix H = K (K + lam I)^-1
    # We need diagonal of H for LOOCV formula
    # H = K @ inv(A)
    # Instead of full inverse, we can solve A X = K to get X = A^-1 K, then H = K X? No.
    # H = K A^-1.
    # Let's just compute inv(A) since N=200 is small.
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.inf
        
    H_diag = np.sum(K * A_inv, axis=1) # Diagonal of K @ A_inv
    
    # LOOCV prediction: y_i - (y_i - y_pred_i) / (1 - H_ii)
    # Residual r_i = y_i - y_pred_i
    # y_loocv_i = y_pred_i - r_i / (1 - H_ii)
    # Error_i = y_i - y_loocv_i = y_i - y_pred_i + r_i / (1 - H_ii) = r_i + r_i / (1 - H_ii) = r_i * (1 + 1/(1-H_ii)) = r_i * (2-H_ii)/(1-H_ii)
    # Wait, standard formula:
    # y_loocv_i = y_i - (y_i - y_pred_i) / (1 - H_ii)
    # Error = y_i - y_loocv_i = (y_i - y_pred_i) / (1 - H_ii)
    # Let's verify.
    # y_pred = H y.
    # y_loocv_i = y_i - (y_i - (Hy)_i) / (1 - H_ii)
    # Error = y_i - y_loocv_i = (y_i - (Hy)_i) / (1 - H_ii)
    # So MSE_LOOCV = mean( (resid / (1 - H_diag))^2 )
    
    resid = y_train - y_pred
    denom = 1.0 - H_diag
    # Guard against division by zero
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    
    loocv_errors = resid / denom
    return np.mean(loocv_errors ** 2)


def fit(X_train, y_train):
    x = X_train[:, 0].astype(np.float64)
    y = y_train.astype(np.float64)
    N = x.shape[0]
    
    # Normalize x to [0, 1] for stable kernel bandwidth
    x_min = np.min(x)
    x_max = np.max(x)
    x_span = x_max - x_min
    if x_span < 1e-10:
        x_span = 1.0
    x_norm = (x - x_min) / x_span
    
    # Grid search parameters
    # Sigma in normalized units. 
    # Small sigma -> local fits, large sigma -> smooth global fits.
    sigma_grid = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
    lam_grid = np.logspace(-5, 5, 21)
    
    best_score = np.inf
    best_sigma = 0.5
    best_lam = 1.0
    best_alpha = None
    best_x_norm = x_norm
    best_x_min = x_min
    best_x_span = x_span
    
    for sigma in sigma_grid:
        if _time_left() < 5.0:
            break
        for lam in lam_grid:
            score = _loocv_rbf(x_norm, y, sigma, lam)
            if score < best_score:
                best_score = score
                best_sigma = sigma
                best_lam = lam
    
    # Final solve with best parameters
    K = _rbf_kernel(x_norm, x_norm, best_sigma)
    A = K + best_lam * np.eye(N)
    try:
        best_alpha = np.linalg.solve(A, y)
    except np.linalg.LinAlgError:
        # Fallback to simple linear regression if KRR fails completely
        # Use polynomial basis as fallback
        X_poly = np.column_stack([np.ones(N), x, x**2, x**3])
        try:
            w_poly = np.linalg.lstsq(X_poly, y, rcond=None)[0]
            def predict_fallback(X):
                X = np.asarray(X, dtype=np.float64)
                if X.ndim == 1: X = X.reshape(1, -1)
                xx = X[:, 0]
                Xp = np.column_stack([np.ones(len(xx)), xx, xx**2, xx**3])
                return Xp @ w_poly
            return predict_fallback
        except:
            # Last resort: mean
            mean_y = np.mean(y)
            def predict_mean(X):
                X = np.asarray(X, dtype=np.float64)
                if X.ndim == 1: X = X.reshape(1, -1)
                return np.full(X.shape[0], mean_y)
            return predict_mean

    # Ensure alpha is finite
    best_alpha = np.where(np.isfinite(best_alpha), best_alpha, 0.0)
    
    # Store parameters for prediction
    pred_x_min = best_x_min
    pred_x_span = best_x_span
    pred_sigma = best_sigma
    pred_alpha = best_alpha
    pred_x_norm_train = x_norm

    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        xx = X[:, 0]
        xx_norm = (xx - pred_x_min) / pred_x_span
        
        # Compute kernel between test points and training points
        K_test_train = _rbf_kernel(xx_norm, pred_x_norm_train, pred_sigma)
        
        out = K_test_train @ pred_alpha
        out = np.where(np.isfinite(out), out, 0.0)
        return out

    return predict
# EVOLVE-BLOCK-END
