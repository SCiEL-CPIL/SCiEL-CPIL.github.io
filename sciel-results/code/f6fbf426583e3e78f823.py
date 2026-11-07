"""Interpolation-based regression for 1D noiseless data with fallbacks."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import PchipInterpolator, CubicSpline, make_interp_spline
import warnings
warnings.filterwarnings("ignore")

# Global time tracking
_START_TIME = None
_TIME_LIMIT = 325.0  # Leave 5s safety margin for 330s limit


def _get_elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME


def _remaining_time():
    return _TIME_LIMIT - _get_elapsed()


def _fit_pchip(X_train, y_train):
    """PCHIP interpolation - monotone cubic that avoids overshoot."""
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    
    # Sort by x
    order = np.argsort(X_train[:, 0])
    x_sorted = X_train[order, 0]
    y_sorted = y_train[order]
    
    # Remove duplicates (keep first)
    unique_mask = np.concatenate([np.ones(1, dtype=bool), np.diff(x_sorted) > 1e-12])
    x_unique = x_sorted[unique_mask]
    y_unique = y_sorted[unique_mask]
    
    pchip = PchipInterpolator(x_unique, y_unique)
    
    def predict(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return pchip(X[:, 0])
    
    return predict


def _fit_cubic_spline(X_train, y_train):
    """Cubic spline interpolation."""
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    
    # Sort by x
    order = np.argsort(X_train[:, 0])
    x_sorted = X_train[order, 0]
    y_sorted = y_train[order]
    
    # Remove duplicates
    unique_mask = np.concatenate([np.ones(1, dtype=bool), np.diff(x_sorted) > 1e-12])
    x_unique = x_sorted[unique_mask]
    y_unique = y_sorted[unique_mask]
    
    try:
        spline = CubicSpline(x_unique, y_unique)
    except Exception:
        return _fit_pchip(X_train, y_train)
    
    def predict(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return spline(X[:, 0])
    
    return predict


def _fit_chebyshev(X_train, y_train, max_degree=20):
    """Chebyshev polynomial fit - good for smooth functions."""
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    
    x = X_train[:, 0]
    
    # Normalize to [-1, 1]
    x_min, x_max = x.min(), x.max()
    x_norm = 2 * (x - x_min) / (x_max - x_min + 1e-12) - 1
    
    # Try different degrees and pick best via cross-validation
    best_degree = 10
    best_score = np.inf
    
    for degree in [8, 10, 12, 15, 18, 20, 25, 30]:
        try:
            # Build Chebyshev basis
            from numpy.polynomial import chebyshev as C
            coeffs = C.chebfit(x_norm, y_train, deg=degree)
            
            # Validate on training data (should be near-zero for noiseless)
            pred_train = C.chebval(x_norm, coeffs)
            train_mse = np.mean((pred_train - y_train) ** 2)
            
            # Simple validation: hold out 10%
            n = len(x)
            n_val = max(10, n // 10)
            val_idx = np.random.default_rng(42).permutation(n)[:n_val]
            val_pred = C.chebval(x_norm[val_idx], coeffs)
            val_mse = np.mean((val_pred - y_train[val_idx]) ** 2)
            
            if val_mse < best_score:
                best_score = val_mse
                best_degree = degree
        except Exception:
            continue
    
    # Final fit with best degree
    from numpy.polynomial import chebyshev as C
    coeffs = C.chebfit(x_norm, y_train, deg=best_degree)
    
    def predict(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        x_pred = X[:, 0]
        x_pred_norm = 2 * (x_pred - x_min) / (x_max - x_min + 1e-12) - 1
        return C.chebval(x_pred_norm, coeffs)
    
    return predict


def _fit_fourier(X_train, y_train, n_terms=15):
    """Fourier series fit - good for periodic or oscillatory functions."""
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    
    x = X_train[:, 0]
    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min + 1e-12
    
    # Build Fourier basis
    cols = [np.ones(len(x))]
    for k in range(1, n_terms + 1):
        cols.append(np.sin(2 * np.pi * k * (x - x_min) / x_range))
        cols.append(np.cos(2 * np.pi * k * (x - x_min) / x_range))
    
    F = np.column_stack(cols)
    
    # Ridge regression for stability
    lambda_val = 1e-6
    w = np.linalg.solve(F.T @ F + lambda_val * np.eye(F.shape[1]), F.T @ y_train)
    
    def features(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        x_pred = X[:, 0]
        cols = [np.ones(len(x_pred))]
        for k in range(1, n_terms + 1):
            cols.append(np.sin(2 * np.pi * k * (x_pred - x_min) / x_range))
            cols.append(np.cos(2 * np.pi * k * (x_pred - x_min) / x_range))
        return np.column_stack(cols)
    
    return lambda X: features(X) @ w


def _fit_kernel_ridge(X_train, y_train, length_scale=0.5, alpha=1e-6):
    """Kernel ridge regression with RBF kernel."""
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    
    x = X_train[:, 0]
    n = len(x)
    
    # RBF kernel matrix
    X1 = x[:, None]
    X2 = x[None, :]
    sq_dist = (X1 - X2) ** 2
    K = np.exp(-sq_dist / (2 * length_scale ** 2))
    
    # Solve (K + alpha*I) w = y
    try:
        w = np.linalg.solve(K + alpha * np.eye(n), y_train)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(K + alpha * np.eye(n), y_train, rcond=None)[0]
    
    def predict(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        x_pred = X[:, 0]
        X1 = x_pred[:, None]
        X2 = x[None, :]
        sq_dist = (X1 - X2) ** 2
        K_s = np.exp(-sq_dist / (2 * length_scale ** 2))
        return K_s @ w
    
    return predict


def _fit_ensemble(X_train, y_train):
    """Try multiple models and select best via validation."""
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    
    n = len(X_train)
    
    # Split for validation
    rng = np.random.default_rng(123)
    indices = rng.permutation(n)
    val_size = max(20, n // 5)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    
    models = []
    
    # Model 1: PCHIP (usually best for 1D noiseless)
    try:
        if _remaining_time() > 5:
            pred_func = _fit_pchip(X_tr, y_tr)
            val_pred = pred_func(X_val)
            if np.all(np.isfinite(val_pred)):
                score = np.mean((val_pred - y_val) ** 2)
                models.append((score, pred_func, 'pchip'))
    except Exception:
        pass
    
    # Model 2: Cubic Spline
    try:
        if _remaining_time() > 5:
            pred_func = _fit_cubic_spline(X_tr, y_tr)
            val_pred = pred_func(X_val)
            if np.all(np.isfinite(val_pred)):
                score = np.mean((val_pred - y_val) ** 2)
                models.append((score, pred_func, 'spline'))
    except Exception:
        pass
    
    # Model 3: Chebyshev
    try:
        if _remaining_time() > 5:
            pred_func = _fit_chebyshev(X_tr, y_tr)
            val_pred = pred_func(X_val)
            if np.all(np.isfinite(val_pred)):
                score = np.mean((val_pred - y_val) ** 2)
                models.append((score, pred_func, 'cheb'))
    except Exception:
        pass
    
    # Model 4: Fourier
    try:
        if _remaining_time() > 5:
            pred_func = _fit_fourier(X_tr, y_tr)
            val_pred = pred_func(X_val)
            if np.all(np.isfinite(val_pred)):
                score = np.mean((val_pred - y_val) ** 2)
                models.append((score, pred_func, 'fourier'))
    except Exception:
        pass
    
    # Model 5: Kernel Ridge
    try:
        if _remaining_time() > 5:
            pred_func = _fit_kernel_ridge(X_tr, y_tr)
            val_pred = pred_func(X_val)
            if np.all(np.isfinite(val_pred)):
                score = np.mean((val_pred - y_val) ** 2)
                models.append((score, pred_func, 'kr'))
    except Exception:
        pass
    
    # Select best model
    if not models:
        # Fallback to simple linear
        w, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
        return lambda X: np.asarray(X) @ w
    
    # Sort by score and return best
    models.sort(key=lambda x: x[0])
    return models[0][1]


def fit(X_train, y_train):
    """Main fit function with time management."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    
    # Try ensemble approach
    try:
        predict_func = _fit_ensemble(X_train, y_train)
    except Exception:
        # Fallback to PCHIP
        try:
            predict_func = _fit_pchip(X_train, y_train)
        except Exception:
            # Final fallback: linear
            w, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
            predict_func = lambda X: np.asarray(X) @ w
    
    # Ensure predictions are finite
    def safe_predict(X):
        try:
            pred = predict_func(X)
            pred = np.asarray(pred, dtype=float)
            # Replace any non-finite values
            pred = np.where(np.isfinite(pred), pred, 0.0)
            return pred
        except Exception:
            X = np.asarray(X, dtype=float)
            return np.zeros(len(X))
    
    return safe_predict
# EVOLVE-BLOCK-END
