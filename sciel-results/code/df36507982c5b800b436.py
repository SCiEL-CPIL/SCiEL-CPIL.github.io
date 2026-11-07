# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

_START_TIME = None
TIME_LIMIT = 300.0
SAFETY_MARGIN = 15.0


def _time_left():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        return TIME_LIMIT - SAFETY_MARGIN
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)


def _build_rich_basis(x):
    """Build a very rich set of basis functions for 1D input."""
    n = len(x)
    cols = [np.ones(n)]
    
    # Polynomial terms up to degree 10
    for k in range(1, 11):
        cols.append(x ** k)
    
    # Trigonometric terms with various frequencies
    for freq in [0.5, 1, 2, 3, 4, 5]:
        cols.append(np.sin(freq * x))
        cols.append(np.cos(freq * x))
    
    # Exponential terms (clipped for stability)
    cols.append(np.exp(np.clip(x, -8, 8)))
    cols.append(np.exp(np.clip(-x, -8, 8)))
    cols.append(np.exp(np.clip(2 * x, -8, 8)))
    cols.append(np.exp(np.clip(-2 * x, -8, 8)))
    
    # Log terms (only where x > 0)
    log_x = np.where(x > 1e-8, np.log(np.maximum(x, 1e-8)), 0.0)
    cols.append(log_x)
    cols.append(log_x ** 2)
    cols.append(log_x ** 3)
    
    # Absolute value and sign
    cols.append(np.abs(x))
    cols.append(np.sign(x))
    
    # Hyperbolic functions
    cols.append(np.tanh(x))
    cols.append(np.tanh(2 * x))
    cols.append(np.tanh(0.5 * x))
    cols.append(np.arctan(x))
    cols.append(np.arctan(2 * x))
    
    # Sqrt and other transforms
    sqrt_x = np.sqrt(np.maximum(np.abs(x), 1e-8))
    cols.append(sqrt_x)
    cols.append(np.sign(x) * sqrt_x)
    
    # Rational-like functions
    cols.append(x / (1 + np.abs(x)))
    cols.append(x / (1 + x ** 2))
    
    return np.column_stack(cols)


def _rbf_kernel(X, Y, gamma=1.0):
    """Compute RBF kernel between X and Y."""
    X2 = np.sum(X ** 2, axis=1)[:, np.newaxis]
    Y2 = np.sum(Y ** 2, axis=1)[np.newaxis, :]
    dist = X2 + Y2 - 2 * X @ Y.T
    dist = np.maximum(dist, 0)
    return np.exp(-gamma * dist)


def _fit_ridge_kernel(X, y, gamma, lam):
    """Fit ridge regression in RBF kernel space."""
    K = _rbf_kernel(X, X, gamma)
    n = len(X)
    try:
        w = np.linalg.solve(K + lam * np.eye(n), y)
        return w
    except np.linalg.LinAlgError:
        w, *_ = np.linalg.lstsq(K + lam * np.eye(n), y, rcond=None)
        return w


def _predict_ridge_kernel(X_train, X_test, y_train, w, gamma):
    """Predict using RBF kernel ridge regression weights."""
    K = _rbf_kernel(X_test, X_train, gamma)
    return K @ w


def _fit_basis_ridge(X, y, lam):
    """Fit ridge regression on rich basis features."""
    Xf = _build_rich_basis(X)
    n_features = Xf.shape[1]
    try:
        A = Xf.T @ Xf
        b = Xf.T @ y
        reg = A + lam * np.eye(n_features)
        w = np.linalg.solve(reg, b)
        return w
    except np.linalg.LinAlgError:
        w, *_ = np.linalg.lstsq(Xf, y, rcond=None)
        return w


def _predict_basis(X, w):
    """Predict using basis ridge regression weights."""
    Xf = _build_rich_basis(X)
    return Xf @ w


def fit(X_train, y_train):
    global _START_TIME
    _START_TIME = time.time()
    
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    
    n, d = X_train.shape
    
    # If input is already normalized or in a reasonable range, use as is
    # Otherwise, we might want to normalize
    x_min = X_train.min(axis=0)
    x_max = X_train.max(axis=0)
    x_range = x_max - x_min
    x_range[x_range < 1e-8] = 1.0
    X_norm = (X_train - x_min) / x_range * 2 - 1  # Scale to [-1, 1]
    
    # Try multiple strategies and pick the best
    best_score = np.inf
    best_predict_fn = None
    
    # Strategy 1: Rich basis + ridge regression
    basis_lambdas = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]
    for lam in basis_lambdas:
        if _time_left() < 5:
            break
        w = _fit_basis_ridge(X_norm[:, 0], y_train, lam)
        
        def predict_basis_fn(X, w=w):
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            Xn = (X - x_min) / x_range * 2 - 1
            return _predict_basis(Xn[:, 0], w)
        
        pred = predict_basis_fn(X_train)
        mse = np.mean((pred - y_train) ** 2)
        var_y = np.var(y_train)
        if var_y > 1e-10:
            nrmse = mse / var_y
        else:
            nrmse = mse
        if nrmse < best_score:
            best_score = nrmse
            best_predict_fn = predict_basis_fn
    
    # Strategy 2: RBF kernel ridge regression with various gammas
    gammas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    kernel_lambdas = [1e-6, 1e-4, 1e-2, 1e-1, 1.0]
    for gamma in gammas:
        if _time_left() < 5:
            break
        for lam in kernel_lambdas:
            if _time_left() < 2:
                break
            w = _fit_ridge_kernel(X_norm, y_train, gamma, lam)
            
            def predict_kernel_fn(X, w=w, gamma=gamma):
                X = np.asarray(X, dtype=np.float64)
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                Xn = (X - x_min) / x_range * 2 - 1
                return _predict_ridge_kernel(X_norm, Xn, y_train, w, gamma)
            
            pred = predict_kernel_fn(X_train)
            mse = np.mean((pred - y_train) ** 2)
            var_y = np.var(y_train)
            if var_y > 1e-10:
                nrmse = mse / var_y
            else:
                nrmse = mse
            if nrmse < best_score:
                best_score = nrmse
                best_predict_fn = predict_kernel_fn
    
    # Strategy 3: Polynomial fit with high degree
    for deg in [3, 5, 7, 9, 11]:
        if _time_left() < 2:
            break
        try:
            coeffs = np.polyfit(X_norm[:, 0], y_train, deg)
            
            def predict_poly_fn(X, coeffs=coeffs):
                X = np.asarray(X, dtype=np.float64)
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                Xn = (X - x_min) / x_range * 2 - 1
                return np.polyval(coeffs, Xn[:, 0])
            
            pred = predict_poly_fn(X_train)
            mse = np.mean((pred - y_train) ** 2)
            var_y = np.var(y_train)
            if var_y > 1e-10:
                nrmse = mse / var_y
            else:
                nrmse = mse
            if nrmse < best_score:
                best_score = nrmse
                best_predict_fn = predict_poly_fn
        except Exception:
            continue
    
    # Strategy 4: Spline interpolation
    if _time_left() > 5:
        try:
            from scipy.interpolate import UnivariateSpline
            x_sorted = X_norm[:, 0].copy()
            y_sorted = y_train.copy()
            idx = np.argsort(x_sorted)
            x_sorted = x_sorted[idx]
            y_sorted = y_sorted[idx]
            
            for s in [0, 1e-4, 1e-2, 1e-1, 1.0]:
                if _time_left() < 2:
                    break
                try:
                    spline = UnivariateSpline(x_sorted, y_sorted, s=s)
                    
                    def predict_spline_fn(X, spline=spline):
                        X = np.asarray(X, dtype=np.float64)
                        if X.ndim == 1:
                            X = X.reshape(-1, 1)
                        Xn = (X - x_min) / x_range * 2 - 1
                        return spline(Xn[:, 0])
                    
                    pred = predict_spline_fn(X_train)
                    mse = np.mean((pred - y_train) ** 2)
                    var_y = np.var(y_train)
                    if var_y > 1e-10:
                        nrmse = mse / var_y
                    else:
                        nrmse = mse
                    if nrmse < best_score:
                        best_score = nrmse
                        best_predict_fn = predict_spline_fn
                except Exception:
                    continue
        except ImportError:
            pass
    
    # Fallback: simple linear regression
    if best_predict_fn is None:
        def predict_linear(X):
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            Xn = (X - x_min) / x_range * 2 - 1
            x = Xn[:, 0]
            a = np.dot(x - x.mean(), y_train - y_train.mean()) / np.dot(x - x.mean(), x - x.mean())
            b = y_train.mean() - a * x.mean()
            return a * x + b
        best_predict_fn = predict_linear
    
    return best_predict_fn
# EVOLVE-BLOCK-END
