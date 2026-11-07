"""LOESS-based regression with adaptive bandwidth via LOOCV and polynomial fallback."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings("ignore")

_START_TIME = None
TIME_LIMIT = 300.0
SAFETY_MARGIN = 20.0

def _time_left():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        return TIME_LIMIT - SAFETY_MARGIN
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)

def _loess_predict(x_new, x_train, y_train, frac, degree=2):
    """
    LOESS prediction at x_new given training data.
    frac: fraction of data points to use (bandwidth)
    degree: polynomial degree for local fit
    """
    x_new = np.asarray(x_new).flatten()
    n = len(x_train)
    k = max(3, int(frac * n))  # number of neighbors
    
    preds = np.zeros(len(x_new))
    for i, xi in enumerate(x_new):
        # Compute distances to all training points
        dists = np.abs(x_train - xi)
        # Get k nearest neighbors
        idx = np.argpartition(dists, k)[:k]
        
        # Tricube weights based on distance
        max_dist = dists[idx].max()
        if max_dist < 1e-12:
            weights = np.ones(k)
        else:
            u = dists[idx] / max_dist
            weights = (1 - u ** 3) ** 3
            weights[weights < 1e-12] = 0
        
        # Build local polynomial design matrix
        x_local = x_train[idx]
        # Center and scale for numerical stability
        xc = x_local - xi
        xs = xc / (max_dist + 1e-12)
        
        cols = [np.ones(k)]
        for d in range(1, degree + 1):
            cols.append(xs ** d)
        X_local = np.column_stack(cols)
        
        # Weighted least squares
        W = np.diag(weights)
        try:
            XtWX = X_local.T @ W @ X_local
            Xty = X_local.T @ W @ y_train[idx]
            # Add small ridge for stability
            XtWX += 1e-8 * np.eye(degree + 1)
            beta = np.linalg.solve(XtWX, Xty)
            # Predict at xi (which is 0 in centered/scaled coords)
            preds[i] = beta[0]
        except Exception:
            # Fallback: weighted mean
            if weights.sum() > 1e-12:
                preds[i] = np.sum(weights * y_train[idx]) / np.sum(weights)
            else:
                preds[i] = y_train[idx][0]
    
    return preds

def _loocv_mse(x_train, y_train, frac, degree=2):
    """
    Compute LOOCV MSE for LOESS with given fraction and degree.
    Uses the hat matrix trick for efficiency.
    """
    n = len(x_train)
    k = max(3, int(frac * n))
    
    # For LOOCV, we need to compute leverage values
    # For LOESS, this requires fitting n times, but we can approximate
    # by using a subset or computing directly for small n
    
    if n > 200:
        # For larger n, use a subsample for LOOCV
        idx = np.random.RandomState(42).choice(n, size=200, replace=False)
        x_sub = x_train[idx]
        y_sub = y_train[idx]
        n_sub = len(x_sub)
    else:
        x_sub = x_train
        y_sub = y_train
        n_sub = n
    
    k_sub = max(3, int(frac * n_sub))
    
    # Compute LOOCV predictions
    loocv_preds = np.zeros(n_sub)
    
    for i in range(n_sub):
        if _time_left() < 5.0:
            break
        # Leave out point i
        mask = np.ones(n_sub, dtype=bool)
        mask[i] = False
        x_loo = x_sub[mask]
        y_loo = y_sub[mask]
        
        # Predict point i using LOESS
        try:
            loocv_preds[i] = _loess_predict(x_sub[i], x_loo, y_loo, frac, degree)[0]
        except Exception:
            loocv_preds[i] = y_sub.mean()
    
    mse = np.mean((loocv_preds - y_sub) ** 2)
    return mse

def _poly_features(X, degree):
    """Polynomial features for 1D input."""
    x = X[:, 0]
    cols = [np.ones(len(x))]
    for k in range(1, degree + 1):
        cols.append(x ** k)
    return np.column_stack(cols)

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
    
    x_train = X_train[:, 0].copy()
    y_train = y_train.copy()
    n = len(x_train)
    
    # Sort by x for efficiency
    sort_idx = np.argsort(x_train)
    x_sorted = x_train[sort_idx]
    y_sorted = y_train[sort_idx]
    
    # Candidate fractions (bandwidths)
    fracs = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    # LOOCV to select best fraction
    best_frac = 0.3
    best_loocv_mse = float('inf')
    
    for frac in fracs:
        if _time_left() < 10.0:
            break
        try:
            loocv_mse = _loocv_mse(x_sorted, y_sorted, frac, degree=2)
            if loocv_mse < best_loocv_mse:
                best_loocv_mse = loocv_mse
                best_frac = frac
        except Exception:
            continue
    
    # Also try degree 1 and 3 for the best fraction
    for deg in [1, 3]:
        if _time_left() < 10.0:
            break
        try:
            loocv_mse = _loocv_mse(x_sorted, y_sorted, best_frac, degree=deg)
            if loocv_mse < best_loocv_mse:
                best_loocv_mse = loocv_mse
                # Keep best_frac, change degree
        except Exception:
            continue
    
    # Fit final LOESS model with best fraction
    try:
        # Use the best fraction for final prediction
        final_frac = best_frac
        final_degree = 2  # Default, could be adjusted
        
        def predict(X):
            x_new = X[:, 0].copy()
            return _loess_predict(x_new, x_sorted, y_sorted, final_frac, final_degree)
        
        # Verify prediction works
        test_pred = predict(X_train[:5])
        if not np.all(np.isfinite(test_pred)):
            raise ValueError("Non-finite predictions")
        
        return predict
    
    except Exception:
        # Fallback to polynomial
        try:
            Xf = _poly_features(X_train, 5)
            w, *_ = np.linalg.lstsq(Xf, y_train, rcond=None)
            return lambda X, w=w: _poly_features(X, 5) @ w
        except Exception:
            # Ultimate fallback: mean
            y_mean = np.mean(y_train)
            return lambda X: np.full(len(X), y_mean)
# EVOLVE-BLOCK-END
