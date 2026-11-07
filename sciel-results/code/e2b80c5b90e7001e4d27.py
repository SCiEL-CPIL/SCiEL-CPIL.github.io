import numpy as np
import time

_START_TIME = None
TIME_LIMIT = 330.0
SAFETY_MARGIN = 30.0

def _time_left():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        return TIME_LIMIT - SAFETY_MARGIN
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)

def _matern52_kernel(X1, X2, length_scale):
    """Compute Matérn-5/2 kernel between 1D arrays X1 and X2."""
    X1 = np.asarray(X1, dtype=np.float64).reshape(-1, 1)
    X2 = np.asarray(X2, dtype=np.float64).reshape(-1, 1)
    d = np.abs(X1 - X2.T) / length_scale
    # Matérn-5/2: (1 + sqrt(5)*d + 5/3*d^2) * exp(-sqrt(5)*d)
    sqrt5 = np.sqrt(5.0)
    K = (1.0 + sqrt5 * d + (5.0/3.0) * d**2) * np.exp(-sqrt5 * d)
    return K

def _gp_predict(X_train, y_train, X_test, length_scale, signal_var, noise_var):
    """Gaussian process prediction with Matérn-5/2 kernel."""
    K = _matern52_kernel(X_train, X_train, length_scale)
    K[np.diag_indices_from(K)] += noise_var
    K_test = _matern52_kernel(X_test, X_train, length_scale)
    
    try:
        # Solve (K + noise*I) alpha = y
        alpha = np.linalg.solve(K, y_train)
        mean = K_test @ alpha
        # For prediction, we only need the mean
        return mean
    except np.linalg.LinAlgError:
        # Fallback to least squares if singular
        try:
            alpha = np.linalg.lstsq(K, y_train, rcond=None)[0]
            mean = K_test @ alpha
            return mean
        except Exception:
            return np.full(len(X_test), np.mean(y_train))

def fit(X_train, y_train):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Ensure X is 2D
    X = np.asarray(X_train, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    n = len(y)
    
    # Extract 1D input
    x = X[:, 0].copy()
    
    # Standardize x for better numerical conditioning
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-12:
        sigma = 1.0
    x_scaled = (x - mu) / sigma
    
    # 5-fold cross-validation for hyperparameter selection
    np.random.seed(42)
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, 5)
    
    # Grid search over length_scale and signal_var
    # length_scale in scaled space: 0.1 to 10
    length_scales = np.logspace(-1, 1, 8)
    # signal_var: 0.1 to 10
    signal_vars = np.logspace(-1, 1, 5)
    # noise_var: small, 1e-6 to 1e-2
    noise_vars = np.logspace(-6, -2, 3)
    
    best_val_mse = np.inf
    best_params = None
    
    # Precompute fold indices
    for ls in length_scales:
        if _time_left() < 15.0:
            break
        for sv in signal_vars:
            if _time_left() < 15.0:
                break
            for nv in noise_vars:
                if _time_left() < 15.0:
                    break
                try:
                    fold_mses = []
                    for i in range(5):
                        val_idx = folds[i]
                        train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
                        
                        x_tr = x_scaled[train_idx]
                        y_tr = y[train_idx]
                        x_val = x_scaled[val_idx]
                        y_val = y[val_idx]
                        
                        pred_val = _gp_predict(x_tr, y_tr, x_val, ls, sv, nv)
                        mse = np.mean((pred_val - y_val)**2)
                        fold_mses.append(mse)
                    
                    mean_mse = np.mean(fold_mses)
                    if mean_mse < best_val_mse:
                        best_val_mse = mean_mse
                        best_params = (ls, sv, nv)
                except Exception:
                    continue
    
    # Fallback if no valid parameters found
    if best_params is None:
        # Use polynomial regression as fallback
        try:
            # Fit a degree 5 polynomial
            coeffs = np.polyfit(x_scaled, y, 5)
            def predict(X):
                X_arr = np.asarray(X, dtype=np.float64)
                if X_arr.ndim == 1:
                    X_arr = X_arr.reshape(-1, 1)
                xv = (X_arr[:, 0] - mu) / sigma
                return np.polyval(coeffs, xv)
        except Exception:
            mean_y = np.mean(y)
            def predict(X):
                return np.full(len(np.asarray(X, dtype=np.float64).reshape(-1)), mean_y)
        return predict
    
    ls_best, sv_best, nv_best = best_params
    
    # Refit on full training set
    # For final prediction, we need to solve the GP system on all data
    K_full = _matern52_kernel(x_scaled, x_scaled, ls_best)
    K_full[np.diag_indices_from(K_full)] += nv_best
    
    try:
        alpha_full = np.linalg.solve(K_full, y)
    except np.linalg.LinAlgError:
        alpha_full = np.linalg.lstsq(K_full, y, rcond=None)[0]
    
    # Store for prediction
    x_train_scaled_store = x_scaled.copy()
    alpha_store = alpha_full.copy()
    ls_store = ls_best
    mu_store = mu
    sigma_store = sigma
    
    def predict(X):
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        xv = (X_arr[:, 0] - mu_store) / sigma_store
        K_test = _matern52_kernel(xv, x_train_scaled_store, ls_store)
        return K_test @ alpha_store
    
    return predict
