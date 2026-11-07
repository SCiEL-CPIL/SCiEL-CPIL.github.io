"""Improved portfolio-based regression with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings("ignore")

_START_TIME = None
TIME_LIMIT = 300.0
SAFETY_MARGIN = 10.0

def _time_left():
    global _START_TIME
    if _START_TIME is None:
        return TIME_LIMIT - SAFETY_MARGIN
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)

def _poly_features(X, degree):
    """Polynomial features for 1D input."""
    x = X[:, 0]
    cols = [np.ones(len(x))]
    for k in range(1, degree + 1):
        cols.append(x ** k)
    return np.column_stack(cols)

def _trig_features(X, n_harmonics=5):
    """Trigonometric features for 1D input."""
    x = X[:, 0]
    cols = [np.ones(len(x))]
    for k in range(1, n_harmonics + 1):
        cols.append(np.sin(k * x))
        cols.append(np.cos(k * x))
    return np.column_stack(cols)

def _mixed_features(X, poly_deg=3, n_harmonics=3):
    """Mixed polynomial and trigonometric features."""
    x = X[:, 0]
    cols = [np.ones(len(x))]
    for k in range(1, poly_deg + 1):
        cols.append(x ** k)
    for k in range(1, n_harmonics + 1):
        cols.append(np.sin(k * x))
        cols.append(np.cos(k * x))
    return np.column_stack(cols)

def _rational_features(X, num_deg=2, den_deg=1):
    """Rational function features: p(x)/q(x) expanded."""
    x = X[:, 0]
    cols = [np.ones(len(x))]
    # Numerator terms
    for k in range(1, num_deg + 1):
        cols.append(x ** k)
    # Cross terms with denominator
    for k in range(1, den_deg + 1):
        cols.append(x ** k / (1 + x ** 2))  # Regularized denominator
    return np.column_stack(cols)

def _exp_features(X):
    """Exponential and log features."""
    x = X[:, 0]
    cols = [np.ones(len(x)), x, x ** 2]
    # Safe exponential (clip to avoid overflow)
    x_clipped = np.clip(x, -10, 10)
    cols.append(np.exp(x_clipped))
    cols.append(np.exp(-x_clipped))
    # Log features (safe)
    x_safe = np.where(np.abs(x) > 1e-6, x, 1e-6)
    cols.append(np.log(np.abs(x_safe)))
    return np.column_stack(cols)

def _fit_model(features_func, X_train, y_train, max_degree=None):
    """Fit a linear model on given features, return (mse, predict_func)."""
    try:
        if max_degree is not None and hasattr(features_func, '__name__'):
            if 'poly' in features_func.__name__:
                Xf = features_func(X_train, max_degree)
            else:
                Xf = features_func(X_train)
        else:
            Xf = features_func(X_train)
        
        w, residuals, rank, sv = np.linalg.lstsq(Xf, y_train, rcond=None)
        y_pred = Xf @ w
        mse = np.mean((y_pred - y_train) ** 2)
        
        def predict(X):
            if max_degree is not None and hasattr(features_func, '__name__'):
                if 'poly' in features_func.__name__:
                    Xf_test = features_func(X, max_degree)
                else:
                    Xf_test = features_func(X)
            else:
                Xf_test = features_func(X)
            return Xf_test @ w
        
        return mse, predict
    except Exception:
        return float('inf'), None

def fit(X_train, y_train):
    global _START_TIME
    _START_TIME = time.time()
    
    d = X_train.shape[1]
    if d != 1:
        # Fallback to simple polynomial for multi-dimensional
        def features(X):
            cols = [np.ones(len(X))]
            for j in range(X.shape[1]):
                cols.append(X[:, j])
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    cols.append(X[:, i] * X[:, j])
            return np.column_stack(cols)
        w, *_ = np.linalg.lstsq(features(X_train), y_train, rcond=None)
        return lambda X: features(X) @ w
    
    # Portfolio of candidate models
    candidates = []
    
    # Polynomial models of various degrees
    for deg in [2, 3, 4, 5, 6, 7, 8]:
        candidates.append(('poly', deg))
    
    # Trigonometric models
    for n in [3, 5, 7, 10]:
        candidates.append(('trig', n))
    
    # Mixed models
    for p, t in [(2,2), (3,3), (4,4)]:
        candidates.append(('mixed', (p, t)))
    
    # Rational models
    for nd, dd in [(2,1), (3,1), (3,2)]:
        candidates.append(('rational', (nd, dd)))
    
    # Exponential models
    candidates.append(('exp', None))
    
    # Fit all candidates and select best
    best_mse = float('inf')
    best_predict = None
    
    for model_type, param in candidates:
        if _time_left() < 5.0:
            break
        
        if model_type == 'poly':
            mse, predict = _fit_model(_poly_features, X_train, y_train, max_degree=param)
        elif model_type == 'trig':
            mse, predict = _fit_model(_trig_features, X_train, y_train)
            # Adjust for number of harmonics
            if predict is not None:
                # Re-fit with specific n_harmonics
                def trig_fit(X, n=param):
                    return _trig_features(X, n_harmonics=n)
                Xf = trig_fit(X_train)
                w, *_ = np.linalg.lstsq(Xf, y_train, rcond=None)
                y_pred = Xf @ w
                mse = np.mean((y_pred - y_train) ** 2)
                predict = lambda X, w=w, n=param: _trig_features(X, n_harmonics=n) @ w
        elif model_type == 'mixed':
            p, t = param
            def mixed_fit(X, p=p, t=t):
                return _mixed_features(X, poly_deg=p, n_harmonics=t)
            Xf = mixed_fit(X_train)
            w, *_ = np.linalg.lstsq(Xf, y_train, rcond=None)
            y_pred = Xf @ w
            mse = np.mean((y_pred - y_train) ** 2)
            predict = lambda X, w=w, p=p, t=t: _mixed_features(X, poly_deg=p, n_harmonics=t) @ w
        elif model_type == 'rational':
            nd, dd = param
            def rat_fit(X, nd=nd, dd=dd):
                return _rational_features(X, num_deg=nd, den_deg=dd)
            Xf = rat_fit(X_train)
            w, *_ = np.linalg.lstsq(Xf, y_train, rcond=None)
            y_pred = Xf @ w
            mse = np.mean((y_pred - y_train) ** 2)
            predict = lambda X, w=w, nd=nd, dd=dd: _rational_features(X, num_deg=nd, den_deg=dd) @ w
        elif model_type == 'exp':
            Xf = _exp_features(X_train)
            w, *_ = np.linalg.lstsq(Xf, y_train, rcond=None)
            y_pred = Xf @ w
            mse = np.mean((y_pred - y_train) ** 2)
            predict = lambda X, w=w: _exp_features(X) @ w
        else:
            continue
        
        if mse < best_mse and predict is not None:
            best_mse = mse
            best_predict = predict
    
    # If no model worked, fallback to degree-3 polynomial
    if best_predict is None:
        Xf = _poly_features(X_train, 3)
        w, *_ = np.linalg.lstsq(Xf, y_train, rcond=None)
        best_predict = lambda X, w=w: _poly_features(X, 3) @ w
    
    return best_predict
# EVOLVE-BLOCK-END
