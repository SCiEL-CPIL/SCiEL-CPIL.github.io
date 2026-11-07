# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
from scipy.optimize import differential_evolution, minimize
from scipy.interpolate import make_interp_spline
import sys

warnings.filterwarnings("ignore")

# Global deadline management
# The task states the time limit covers all equations together.
# We assume the module is loaded once and fit is called multiple times.
_START_TIME = time.time()
_TOTAL_TIME_LIMIT = 325.0  # Safety margin under 330s

def _get_remaining_time():
    return _TOTAL_TIME_LIMIT - (time.time() - _START_TIME)

def _nrmse(pred, y):
    """Normalized MSE: mean((pred-y)^2) / var(y)"""
    var_y = np.var(y)
    if var_y < 1e-15:
        return 0.0
    return np.mean((pred - y) ** 2) / var_y

def _prepare_data(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    return X_train, y_train

def _fit_linear_basis(x, y, features_func, alpha=1e-6):
    """Fit a linear model y = F(x) @ w using Ridge Regression."""
    try:
        F = features_func(x)
        # Add small ridge for stability
        A = F.T @ F + alpha * np.eye(F.shape[1])
        b = F.T @ y
        w = np.linalg.solve(A, b)
        pred = F @ w
        return pred, w, F.shape[1]
    except Exception:
        return None, None, 0

def _features_poly(x, deg):
    cols = [np.ones_like(x)]
    for d in range(1, deg + 1):
        cols.append(x ** d)
    return np.column_stack(cols)

def _features_trig(x, n_harm):
    cols = [np.ones_like(x)]
    for k in range(1, n_harm + 1):
        cols.append(np.sin(k * x))
        cols.append(np.cos(k * x))
    return np.column_stack(cols)

def _features_rbf(x, centers, bandwidths):
    cols = [np.ones_like(x)]
    for c, b in zip(centers, bandwidths):
        cols.append(np.exp(-((x - c) ** 2) / (2 * b ** 2)))
    return np.column_stack(cols)

def _fit_parametric_model(x, y, model_func, bounds, time_budget=10):
    """
    Fit a parametric model y = model_func(x, params) using differential evolution.
    model_func(x, params) should return predictions.
    """
    n_params = len(bounds)
    
    def objective(params):
        try:
            pred = model_func(x, params)
            # Check for NaNs/Infs
            if not np.all(np.isfinite(pred)):
                return 1e10
            return np.mean((pred - y) ** 2)
        except Exception:
            return 1e10

    try:
        # Use differential evolution for global search
        # maxiter and popsize adjusted based on time budget
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=50, 
            popsize=10, 
            tol=1e-5,
            seed=42,
            polish=True
        )
        
        if result.success or result.fun < 1e10:
            best_params = result.x
            pred = model_func(x, best_params)
            return pred, best_params
    except Exception:
        pass
    return None, None

def fit(X_train, y_train):
    X_train, y_train = _prepare_data(X_train, y_train)
    x = X_train[:, 0]
    y = y_train
    
    n = len(x)
    
    # Check time
    remaining = _get_remaining_time()
    if remaining < 5:
        # Fallback: Linear fit
        A = np.column_stack([np.ones(n), x])
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        def predict_fallback(X):
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1: X = X.reshape(1, -1)
            return A @ w
        return predict_fallback

    # Split into train and validation for model selection
    # Since data is noiseless, validation error is a good proxy for test error
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    val_size = 40
    val_idx = indices[:val_size]
    tr_idx = indices[val_size:]
    
    x_tr, y_tr = x[tr_idx], y[tr_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    
    best_score = np.inf
    best_predict_fn = None
    best_name = "none"

    # Helper to update best
    def update_best(pred_tr, pred_val, predict_fn, name):
        nonlocal best_score, best_predict_fn, best_name
        # Calculate validation score
        score = _nrmse(pred_val, y_val)
        if score < best_score:
            best_score = score
            best_predict_fn = predict_fn
            best_name = name
            # Optionally print for debugging if needed, but keep silent for speed
            # print(f"New best: {name}, score: {score:.6f}")

    # 1. Polynomial Basis
    for deg in [1, 2, 3, 4, 5, 6, 8, 10]:
        if _get_remaining_time() < 10: break
        pred_tr, w, _ = _fit_linear_basis(x_tr, y_tr, lambda x_: _features_poly(x_, deg))
        if w is not None:
            pred_val = _features_poly(x_val, deg) @ w
            def predict_poly(X, w=w, d=deg):
                X = np.asarray(X, dtype=np.float64)
                if X.ndim == 1: X = X.reshape(1, -1)
                xx = X[:, 0]
                return _features_poly(xx, d) @ w
            update_best(pred_tr, pred_val, predict_poly, f"Poly_{deg}")

    # 2. Trigonometric Basis
    for nh in [3, 5, 8, 12]:
        if _get_remaining_time() < 10: break
        pred_tr, w, _ = _fit_linear_basis(x_tr, y_tr, lambda x_: _features_trig(x_, nh))
        if w is not None:
            pred_val = _features_trig(x_val, nh) @ w
            def predict_trig(X, w=w, n=nh):
                X = np.asarray(X, dtype=np.float64)
                if X.ndim == 1: X = X.reshape(1, -1)
                xx = X[:, 0]
                return _features_trig(xx, n) @ w
            update_best(pred_tr, pred_val, predict_trig, f"Trig_{nh}")

    # 3. RBF Basis
    x_min, x_max = x.min(), x.max()
    for n_centers in [10, 20, 30]:
        centers = np.linspace(x_min, x_max, n_centers)
        for bw in [0.5, 1.0, 2.0]:
            if _get_remaining_time() < 10: break
            pred_tr, w, _ = _fit_linear_basis(x_tr, y_tr, lambda x_: _features_rbf(x_, centers, [bw]*n_centers))
            if w is not None:
                pred_val = _features_rbf(x_val, centers, [bw]*n_centers) @ w
                def predict_rbf(X, w=w, c=centers, b=bw):
                    X = np.asarray(X, dtype=np.float64)
                    if X.ndim == 1: X = X.reshape(1, -1)
                    xx = X[:, 0]
                    return _features_rbf(xx, c, [b]*len(c)) @ w
                update_best(pred_tr, pred_val, predict_rbf, f"RBF_{n_centers}_{bw}")

    # 4. Spline Interpolation
    if _get_remaining_time() > 20:
        try:
            # Sort points for spline
            x_sorted = np.sort(x_tr)
            y_sorted = y_tr[np.argsort(x_tr)]
            
            # Use a subset of points as knots to avoid overfitting
            n_knots = min(len(x_sorted), 20)
            if n_knots >= 5:
                knots = np.linspace(x_sorted[0], x_sorted[-1], n_knots)
                # Interpolate y at knots
                y_at_knots = np.interp(knots, x_sorted, y_sorted)
                spl = make_interp_spline(knots, y_at_knots, k=3)
                
                def predict_spline(X):
                    X = np.asarray(X, dtype=np.float64)
                    if X.ndim == 1: X = X.reshape(1, -1)
                    xx = X[:, 0]
                    return spl(xx)
                
                pred_tr = predict_spline(X_train[tr_idx])
                pred_val = predict_spline(X_train[val_idx])
                update_best(pred_tr, pred_val, predict_spline, "Spline")
        except Exception:
            pass

    # 5. Parametric Symbolic Search
    # This is the most powerful method for finding exact forms
    if _get_remaining_time() > 30:
        # Define candidate models
        models = []
        
        # Sin: A * sin(B * x + C) + D
        def m_sin(x, p):
            return p[0] * np.sin(p[1] * x + p[2]) + p[3]
        bounds_sin = [(-10, 10), (0.1, 5.0), (-np.pi, np.pi), (-10, 10)]
        models.append(("Sin", m_sin, bounds_sin))
        
        # Cos: A * cos(B * x + C) + D
        def m_cos(x, p):
            return p[0] * np.cos(p[1] * x + p[2]) + p[3]
        bounds_cos = [(-10, 10), (0.1, 5.0), (-np.pi, np.pi), (-10, 10)]
        models.append(("Cos", m_cos, bounds_cos))
        
        # Exp: A * exp(B * x) + C
        def m_exp(x, p):
            return p[0] * np.exp(p[1] * x) + p[2]
        bounds_exp = [(-10, 10), (-2.0, 2.0), (-10, 10)]
        models.append(("Exp", m_exp, bounds_exp))
        
        # Log: A * log(x) + B (only if x > 0)
        if x.min() > 0:
            def m_log(x, p):
                return p[0] * np.log(x) + p[1]
            bounds_log = [(-10, 10), (-10, 10)]
            models.append(("Log", m_log, bounds_log))
            
            # Poly Log: A * log(x)^2 + B * log(x) + C
            def m_log2(x, p):
                lx = np.log(x)
                return p[0] * lx**2 + p[1] * lx + p[2]
            bounds_log2 = [(-10, 10), (-10, 10), (-10, 10)]
            models.append(("Log2", m_log2, bounds_log2))

        # Tanh: A * tanh(B * x) + C
        def m_tanh(x, p):
            return p[0] * np.tanh(p[1] * x) + p[2]
        bounds_tanh = [(-10, 10), (0.1, 5.0), (-10, 10)]
        models.append(("Tanh", m_tanh, bounds_tanh))
        
        # Power: A * x^B + C
        def m_power(x, p):
            # Handle negative x for non-integer B
            try:
                return p[0] * np.power(x, p[1]) + p[2]
            except:
                return np.full_like(x, 1e10)
        bounds_power = [(-10, 10), (0.1, 5.0), (-10, 10)]
        models.append(("Power", m_power, bounds_power))
        
        # Gaussian: A * exp(-B * (x-C)^2) + D
        def m_gauss(x, p):
            return p[0] * np.exp(-p[1] * (x - p[2])**2) + p[3]
        bounds_gauss = [(-10, 10), (0.1, 10.0), (x.min(), x.max()), (-10, 10)]
        models.append(("Gauss", m_gauss, bounds_gauss))
        
        # Linear: A * x + B
        def m_lin(x, p):
            return p[0] * x + p[1]
        bounds_lin = [(-10, 10), (-10, 10)]
        models.append(("Lin", m_lin, bounds_lin))
        
        # Quadratic: A * x^2 + B * x + C
        def m_quad(x, p):
            return p[0] * x**2 + p[1] * x + p[2]
        bounds_quad = [(-10, 10), (-10, 10), (-10, 10)]
        models.append(("Quad", m_quad, bounds_quad))

        for name, func, bounds in models:
            if _get_remaining_time() < 15: break
            
            # Use training data for fitting, but we can use all data for final verification
            # Differential evolution is slow, so we limit it.
            # We fit on x_tr, y_tr to get parameters, then evaluate on x_val
            pred_tr, params = _fit_parametric_model(x_tr, y_tr, func, bounds, time_budget=10)
            
            if params is not None:
                # Evaluate on validation set
                pred_val = func(x_val, params)
                
                if np.all(np.isfinite(pred_val)):
                    def predict_param(X, p=params, f=func):
                        X = np.asarray(X, dtype=np.float64)
                        if X.ndim == 1: X = X.reshape(1, -1)
                        xx = X[:, 0]
                        return f(xx, p)
                    
                    update_best(pred_tr, pred_val, predict_param, f"Param_{name}")

    # If no model was found (unlikely), fallback to linear
    if best_predict_fn is None:
        A = np.column_stack([np.ones(n), x])
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        def predict_final(X):
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1: X = X.reshape(1, -1)
            return A @ w
        return predict_final

    # Final check: ensure the chosen model works on the full training set
    # and returns finite values
    try:
        # Quick sanity check on a few points
        X_check = X_train[:5]
        y_check = best_predict_fn(X_check)
        if not np.all(np.isfinite(y_check)):
            raise ValueError("Non-finite predictions")
    except Exception:
        # Fallback to linear if best model fails
        A = np.column_stack([np.ones(n), x])
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        def predict_final(X):
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1: X = X.reshape(1, -1)
            return A @ w
        return predict_final

    def safe_predict(X):
        try:
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            y = best_predict_fn(X)
            y = np.asarray(y, dtype=np.float64)
            y = np.nan_to_num(y, nan=0.0, posinf=1e10, neginf=-1e10)
            return y
        except Exception:
            return np.full(len(X), np.mean(y_train), dtype=np.float64)

    return safe_predict
# EVOLVE-BLOCK-END
