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

def _chebyshev_terms(x, n):
    """Compute Chebyshev polynomial values T_0(x), T_1(x), ..., T_{n-1}(x)."""
    x = np.asarray(x, dtype=np.float64)
    if n <= 0:
        return np.empty((len(x), 0))
    if n == 1:
        return np.ones((len(x), 1))
    if n == 2:
        return np.stack([np.ones_like(x), x], axis=1)
    
    T = np.empty((len(x), n))
    T[:, 0] = 1.0
    T[:, 1] = x
    for k in range(2, n):
        T[:, k] = 2.0 * x * T[:, k-1] - T[:, k-2]
    return T

def _build_rich_basis(x, n_cheb, n_fourier, x_min, x_max):
    """Build a rich basis: Chebyshev + Fourier terms."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    
    # Chebyshev basis (map x to [-1, 1])
    if x_max > x_min:
        x_norm = (x - x_min) / (x_max - x_min) * 2.0 - 1.0
    else:
        x_norm = np.zeros_like(x)
    cheb = _chebyshev_terms(x_norm, n_cheb)
    
    # Fourier basis
    if n_fourier > 0:
        freqs = np.arange(1, n_fourier + 1)
        # Use period based on range
        period = x_max - x_min if x_max > x_min else 1.0
        angles = 2.0 * np.pi * np.outer(x, freqs) / period
        sin_terms = np.sin(angles)
        cos_terms = np.cos(angles)
        fourier = np.hstack([sin_terms, cos_terms])
    else:
        fourier = np.empty((n, 0))
    
    return np.hstack([cheb, fourier])

def _loocv_ridge(X, y, basis, lambdas):
    """LOOCV for ridge regression using the hat matrix trick."""
    n = len(y)
    best_lambda = lambdas[0]
    best_mse = np.inf
    
    for lam in lambdas:
        A = basis.T @ basis + lam * np.eye(basis.shape[1])
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            continue
        H = basis @ A_inv @ basis.T
        diag = np.diag(H)
        
        # LOOCV residuals
        residuals = (y - basis @ (A_inv @ (basis.T @ y))) / (1.0 - diag)
        mse = np.mean(residuals**2)
        
        if mse < best_mse:
            best_mse = mse
            best_lambda = lam
    
    return best_lambda, best_mse

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
    x_min, x_max = np.min(x), np.max(x)
    
    # Store original range for prediction
    x_range = x_max - x_min if x_max > x_min else 1.0
    
    # Try multiple approaches and pick the best via LOOCV
    
    # Approach 1: Rich basis (Chebyshev + Fourier) with ridge
    best_score = np.inf
    best_predict_fn = None
    
    # Grid of basis sizes
    n_cheb_options = [8, 12, 16, 20, 25, 30]
    n_fourier_options = [0, 5, 10, 15, 20]
    lambda_options = np.logspace(-4, 2, 15)
    
    for n_cheb in n_cheb_options:
        if _time_left() < 10.0:
            break
        for n_fourier in n_fourier_options:
            if _time_left() < 10.0:
                break
            try:
                basis = _build_rich_basis(x, n_cheb, n_fourier, x_min, x_max)
                if basis.shape[1] >= n:
                    continue  # Too many basis functions
                
                best_lambda, best_mse = _loocv_ridge(x, y, basis, lambda_options)
                
                if best_mse < best_score:
                    # Fit with best lambda
                    A = basis.T @ basis + best_lambda * np.eye(basis.shape[1])
                    try:
                        weights = np.linalg.solve(A, basis.T @ y)
                    except np.linalg.LinAlgError:
                        weights = np.linalg.lstsq(A, basis.T @ y, rcond=None)[0]
                    
                    best_score = best_mse
                    # Store basis parameters for prediction
                    best_n_cheb = n_cheb
                    best_n_fourier = n_fourier
                    best_weights = weights.copy()
                    best_x_min = x_min
                    best_x_max = x_max
                    best_x_range = x_range
            except Exception:
                continue
    
    # Approach 2: GP with RBF kernel
    if _time_left() > 30.0:
        def _rbf_kernel(X1, X2, length_scale):
            X1 = np.asarray(X1, dtype=np.float64).reshape(-1, 1)
            X2 = np.asarray(X2, dtype=np.float64).reshape(-1, 1)
            d2 = np.sum((X1 - X2.T)**2, axis=1)
            return np.exp(-0.5 * d2 / (length_scale**2))
        
        def _gp_predict_rbf(X_train, y_train, X_test, length_scale, noise_var):
            K = _rbf_kernel(X_train, X_train, length_scale)
            K[np.diag_indices_from(K)] += noise_var
            K_test = _rbf_kernel(X_test, X_train, length_scale)
            try:
                alpha = np.linalg.solve(K, y_train)
                return K_test @ alpha
            except np.linalg.LinAlgError:
                alpha = np.linalg.lstsq(K, y_train, rcond=None)[0]
                return K_test @ alpha
        
        # Simple GP cross-validation
        length_scales = np.logspace(-2, 1, 6)
        noise_vars = np.logspace(-6, -2, 3)
        
        np.random.seed(42)
        indices = np.arange(n)
        np.random.shuffle(indices)
        folds = np.array_split(indices, 5)
        
        best_gp_mse = np.inf
        best_gp_params = None
        
        for ls in length_scales:
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
                        
                        pred_val = _gp_predict_rbf(x[train_idx], y[train_idx], x[val_idx], ls, nv)
                        mse = np.mean((pred_val - y[val_idx])**2)
                        fold_mses.append(mse)
                    
                    mean_mse = np.mean(fold_mses)
                    if mean_mse < best_gp_mse:
                        best_gp_mse = mean_mse
                        best_gp_params = (ls, nv)
                except Exception:
                    continue
        
        if best_gp_params is not None and best_gp_mse < best_score:
            ls_best, nv_best = best_gp_params
            # Refit on full data
            K_full = _rbf_kernel(x, x, ls_best)
            K_full[np.diag_indices_from(K_full)] += nv_best
            try:
                alpha_full = np.linalg.solve(K_full, y)
            except np.linalg.LinAlgError:
                alpha_full = np.linalg.lstsq(K_full, y, rcond=None)[0]
            
            best_score = best_gp_mse
            # Create GP predict function
            x_train_store = x.copy()
            alpha_store = alpha_full.copy()
            ls_store = ls_best
            
            def predict(X):
                X_arr = np.asarray(X, dtype=np.float64)
                if X_arr.ndim == 1:
                    X_arr = X_arr.reshape(-1, 1)
                xv = X_arr[:, 0]
                K_test = _rbf_kernel(xv, x_train_store, ls_store)
                return K_test @ alpha_store
            return predict
    
    # Use best basis approach
    if best_score < np.inf and best_predict_fn is None:
        try:
            basis_store = _build_rich_basis(x, best_n_cheb, best_n_fourier, best_x_min, best_x_max)
            # Already have weights
            def predict(X):
                X_arr = np.asarray(X, dtype=np.float64)
                if X_arr.ndim == 1:
                    X_arr = X_arr.reshape(-1, 1)
                xv = X_arr[:, 0]
                basis_test = _build_rich_basis(xv, best_n_cheb, best_n_fourier, best_x_min, best_x_max)
                return basis_test @ best_weights
            return predict
        except Exception:
            pass
    
    # Fallback: polynomial regression
    try:
        # Fit a degree 7 polynomial
        coeffs = np.polyfit(x, y, 7)
        def predict(X):
            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            return np.polyval(coeffs, X_arr[:, 0])
        return predict
    except Exception:
        mean_y = np.mean(y)
        def predict(X):
            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            return np.full(len(X_arr), mean_y)
        return predict
