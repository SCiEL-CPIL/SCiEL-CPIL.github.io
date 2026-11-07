"""Symbolic regression: ensemble of polynomial and RBF kernels, with global time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

_TOTAL_START = None
_TIME_LIMIT = 325.0  # Safety margin before 330s kill

def _elapsed():
    global _TOTAL_START
    if _TOTAL_START is None:
        _TOTAL_START = time.time()
    return time.time() - _TOTAL_START

def _remaining():
    return max(0.0, _TIME_LIMIT - _elapsed())

def _safe_2d(X):
    """Ensure X is 2D with shape (n, d)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X

def _rbf_kernel(X1, X2, lengthscales, variance):
    """Compute RBF kernel between two sets of points with per-dimension length scales."""
    # X1: (n1, d), X2: (n2, d)
    diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]  # (n1, n2, d)
    sq_dist = np.sum(diff ** 2 / (lengthscales[np.newaxis, np.newaxis, :] ** 2), axis=2)
    return variance * np.exp(-0.5 * sq_dist)

def _make_predict(X_norm, alpha, kernel_fn, y_mean, y_std):
    """Create a predict closure."""
    def predict(X_test):
        X_t = _safe_2d(X_test)
        X_t_norm = (X_t - X_norm[0]) / X_norm[1]
        # kernel_fn returns (n_test, n_train) kernel matrix
        K = kernel_fn(X_t_norm, X_norm[2])  # X_norm[2] is X_train_norm
        y_pred_norm = K @ alpha
        return y_pred_norm * y_std + y_mean
    return predict

def fit(X_train, y_train):
    global _TOTAL_START
    if _TOTAL_START is None:
        _TOTAL_START = time.time()
    
    X = _safe_2d(X_train)
    y = np.asarray(y_train, dtype=np.float64).ravel()
    n, d = X.shape
    
    # Normalize inputs to [0, 1]
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range < 1e-10] = 1.0
    X_norm = (X - X_min) / X_range
    
    # Normalize y
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-10:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    # Strategy 1: Polynomial regression (degree 2-3)
    poly_degree = min(3, max(2, d))  # 2 or 3
    def _poly_features(X):
        """Generate polynomial features up to degree poly_degree."""
        n = X.shape[0]
        d = X.shape[1]
        # Start with bias and linear terms
        cols = [np.ones(n)]
        for j in range(d):
            cols.append(X[:, j])
        # Quadratic and higher terms
        for deg in range(2, poly_degree + 1):
            # All combinations with replacement
            for i1 in range(d):
                for i2 in range(i1, d):
                    if deg == 2:
                        cols.append(X[:, i1] * X[:, i2])
                    elif deg == 3:
                        for i3 in range(i2, d):
                            cols.append(X[:, i1] * X[:, i2] * X[:, i3])
        F = np.column_stack(cols)
        # Add a small ridge to handle overfitting
        return F
    
    F_poly = _poly_features(X_norm)
    n_params_poly = F_poly.shape[1]
    # Ridge regression
    lam_poly = 1e-3
    A = F_poly.T @ F_poly + lam_poly * np.eye(n_params_poly)
    b = F_poly.T @ y_norm
    try:
        w_poly = np.linalg.solve(A, b)
    except:
        w_poly = np.zeros(n_params_poly)
    
    def _predict_poly(X_test):
        X_t = _safe_2d(X_test)
        X_t_norm = (X_t - X_min) / X_range
        F_t = _poly_features(X_t_norm)
        return (F_t @ w_poly) * y_std + y_mean
    
    # Evaluate poly MSE on training data
    pred_poly = _predict_poly(X)
    mse_poly = np.mean((pred_poly - y) ** 2)
    var_y = np.var(y)
    nrmse_poly = mse_poly / (var_y + 1e-15)
    
    # Strategy 2: GP with RBF kernel
    # Try a few sets of length scales
    best_ls = None
    best_nll = np.inf
    best_alpha = None
    best_L = None
    
    n_starts = min(5, max(2, int(_remaining() / 20.0)))
    
    for start_i in range(n_starts):
        if _remaining() < 10.0:
            break
        
        if start_i == 0:
            # Default: moderate length scales
            ls = np.ones(d) * 0.5
            sig2 = 1.0
            noise2 = 0.01
        else:
            np.random.seed(42 + start_i)
            ls = np.random.uniform(0.1, 2.0, size=d)
            sig2 = np.random.uniform(0.1, 5.0)
            noise2 = np.random.uniform(0.001, 0.1)
        
        K = _rbf_kernel(X_norm, X_norm, ls, sig2) + noise2 * np.eye(n)
        try:
            L = np.linalg.cholesky(K + 1e-8 * np.eye(n))
        except np.linalg.LinAlgError:
            continue
        
        try:
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
        except:
            continue
        
        nll = 0.5 * y_norm @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * n * np.log(2 * np.pi)
        
        if nll < best_nll:
            best_nll = nll
            best_ls = ls.copy()
            best_sig2 = sig2
            best_noise2 = noise2
            best_alpha = alpha.copy()
            best_L = L.copy()
    
    # If GP succeeded, evaluate its training MSE
    if best_alpha is not None:
        K_train_test = _rbf_kernel(X_norm, X_norm, best_ls, best_sig2)
        pred_gp_norm = K_train_test @ best_alpha
        pred_gp = pred_gp_norm * y_std + y_mean
        mse_gp = np.mean((pred_gp - y) ** 2)
        nrmse_gp = mse_gp / (var_y + 1e-15)
    else:
        nrmse_gp = np.inf
    
    # Choose the better model
    use_poly = nrmse_poly < nrmse_gp
    
    if use_poly:
        def predict(X_test):
            return _predict_poly(X_test)
    else:
        # Create GP predict
        X_train_norm = X_norm
        kernel_fn = lambda X1, X2: _rbf_kernel(X1, X2, best_ls, best_sig2)
        
        def predict(X_test):
            X_t = _safe_2d(X_test)
            X_t_norm = (X_t - X_min) / X_range
            K = _rbf_kernel(X_t_norm, X_train_norm, best_ls, best_sig2)
            y_pred_norm = K @ best_alpha
            return y_pred_norm * y_std + y_mean
    
    return predict
# EVOLVE-BLOCK-END
