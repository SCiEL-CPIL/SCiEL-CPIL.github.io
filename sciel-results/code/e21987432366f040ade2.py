"""
Strategy: Gaussian Process Regression with Matérn 5/2 kernel and MLL optimization.
This approach treats the hidden equation as a sample from a Gaussian Process prior.
Instead of restricting to a fixed basis (Fourier, Chebyshev, etc.), it learns the
characteristic length scale and signal variance directly from the data by maximizing
the marginal likelihood. This is structurally different from basis expansion because
it adapts the smoothness and correlation structure to the specific equation, handling
polynomials, periodic functions, and irregular smooth functions uniformly without
needing to guess the correct functional family.
"""
import numpy as np
import time
import warnings
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")

_START_TIME = None
TIME_LIMIT = 300.0
SAFETY_MARGIN = 15.0

def _time_left():
    global _START_TIME
    if _START_TIME is None:
        return TIME_LIMIT - SAFETY_MARGIN
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)

def _matern52_kernel(x1, x2, length_scale, signal_var):
    """
    Matérn 5/2 kernel for 1D input.
    k(x1, x2) = signal_var * (1 + sqrt(5)*d + 5/3*d^2) * exp(-sqrt(5)*d)
    where d = |x1 - x2| / length_scale
    """
    d = np.abs(x1[:, None] - x2[None, :]) / length_scale
    sqrt5 = np.sqrt(5.0)
    kernel = signal_var * (1.0 + sqrt5 * d + (5.0 / 3.0) * d**2) * np.exp(-sqrt5 * d)
    return kernel

def _fit_gp(X, y, time_budget):
    """
    Fit a Gaussian Process model.
    Returns a predict function.
    """
    x = X[:, 0]
    n = len(x)
    
    # Initial hyperparameters
    # length_scale: range [0.1, 10]
    # signal_var: range [0.1, 100]
    # noise_var: fixed small value for numerical stability (data is noiseless)
    noise_var = 1e-8
    
    # Transform to log-space for unconstrained optimization
    def _transform(h):
        length_scale = np.exp(h[0])
        signal_var = np.exp(h[1])
        return length_scale, signal_var

    def _neg_log_likelihood(h):
        try:
            if _time_left() < 5:
                return 1e10
            length_scale, signal_var = _transform(h)
            
            # Compute kernel matrix
            K = _matern52_kernel(x, x, length_scale, signal_var)
            K += noise_var * np.eye(n)
            
            # Compute log marginal likelihood
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                # Add jitter if Cholesky fails
                K += 1e-6 * np.eye(n)
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    return 1e10
            
            # alpha = K^{-1} y
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            
            # log p(y|X) = -0.5 * y^T K^{-1} y - 0.5 * log(det(K)) - n/2 * log(2*pi)
            log_det_K = 2.0 * np.sum(np.log(np.diag(L)))
            nll = 0.5 * y.dot(alpha) + 0.5 * log_det_K + 0.5 * n * np.log(2.0 * np.pi)
            
            if not np.isfinite(nll):
                return 1e10
            return nll
        except Exception:
            return 1e10

    # Initial guess: length_scale ~ 1.0, signal_var ~ var(y)
    init_length_scale = 1.0
    init_signal_var = max(np.var(y), 0.1)
    h0 = np.array([np.log(init_length_scale), np.log(init_signal_var)])
    
    # Optimization
    best_h = h0
    best_nll = _neg_log_likelihood(h0)
    
    # Try multiple random starts to avoid local minima
    np.random.seed(42)
    n_starts = 10
    for i in range(n_starts):
        if _time_left() < 10:
            break
        # Perturb initial guess
        h_init = h0 + np.random.randn(2) * 0.5
        # Clip to reasonable log-range
        h_init[0] = np.clip(h_init[0], np.log(0.05), np.log(20.0))
        h_init[1] = np.clip(h_init[1], np.log(0.01), np.log(1000.0))
        
        res = minimize(_neg_log_likelihood, h_init, method='L-BFGS-B', 
                       options={'maxiter': 100, 'ftol': 1e-6})
        
        if res.fun < best_nll:
            best_nll = res.fun
            best_h = res.x
            
    # Final fit with best hyperparameters
    length_scale, signal_var = _transform(best_h)
    
    # Compute alpha for prediction
    K = _matern52_kernel(x, x, length_scale, signal_var)
    K += noise_var * np.eye(n)
    K += 1e-6 * np.eye(n) # Extra jitter for stability
    
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    except np.linalg.LinAlgError:
        # Fallback to pinv
        try:
            alpha = np.linalg.pinv(K) @ y
        except:
            return lambda X_in: np.full(len(X_in), y.mean())
    
    def predict(X_in):
        xi = X_in[:, 0]
        K_star = _matern52_kernel(xi, x, length_scale, signal_var)
        # Mean prediction: K_star @ alpha
        y_pred = K_star @ alpha
        # Ensure finite
        y_pred = np.where(np.isfinite(y_pred), y_pred, y.mean())
        return y_pred
    
    return predict

def _fit_fallback(X, y):
    """
    Fallback: High-degree polynomial via Chebyshev if GP fails or times out.
    """
    x = X[:, 0]
    x_min, x_max = x.min(), x.max()
    if x_max == x_min:
        return lambda X_in: np.full(len(X_in), y.mean())
    x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
    
    degree = 15
    T = np.ones((len(x), degree + 1))
    if degree >= 1:
        T[:, 1] = x_norm
    for n in range(2, degree + 1):
        T[:, n] = 2 * x_norm * T[:, n-1] - T[:, n-2]
        
    w, *_ = np.linalg.lstsq(T, y, rcond=None)
    
    def predict(X_in):
        xi = X_in[:, 0]
        xi_norm = 2 * (xi - x_min) / (x_max - x_min) - 1
        T_in = np.ones((len(xi), degree + 1))
        if degree >= 1:
            T_in[:, 1] = xi_norm
        for n in range(2, degree + 1):
            T_in[:, n] = 2 * xi_norm * T_in[:, n-1] - T_in[:, n-2]
        return T_in @ w
    return predict

def fit(X_train, y_train):
    global _START_TIME
    _START_TIME = time.time()
    
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    
    # Ensure finite
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    
    if len(X) < 5:
        return lambda X_in: np.full(len(X_in), y.mean())
        
    try:
        if _time_left() > 30:
            predict_fn = _fit_gp(X, y, _time_left())
            # Check if prediction is valid
            test_pred = predict_fn(X)
            if np.all(np.isfinite(test_pred)):
                # Quick sanity check: if GP prediction is wildly off, fall back
                mse = np.mean((test_pred - y)**2)
                var_y = np.var(y)
                if var_y > 0 and mse / var_y > 100:
                    # GP might have failed to converge or found a bad local min
                    pass
                else:
                    return predict_fn
    except Exception:
        pass
    
    # Fallback
    try:
        if _time_left() > 10:
            return _fit_fallback(X, y)
    except:
        pass
        
    # Final fallback: constant
    y_mean = y.mean()
    return lambda X_in: np.full(len(X_in), y_mean)
