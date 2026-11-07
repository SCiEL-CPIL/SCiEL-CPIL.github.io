"""
Strategy: Ensemble of diverse basis-function regressors with grid-search hyperparameter tuning.
Unlike a single GP, this approach uses multiple functional forms (Fourier, Chebyshev, RBF, Spline)
to capture different types of hidden equations (periodic, polynomial, localized, smooth).
Hyperparameters are selected via a coarse-to-fine grid search, which is more robust than
gradient-based methods for small datasets and avoids local minima.
The final prediction is a weighted average of the ensemble members based on their CV scores.
"""
import numpy as np
import time
import warnings
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.interpolate import UnivariateSpline

warnings.filterwarnings("ignore")

_START_TIME = None
TIME_LIMIT = 300.0
SAFETY_MARGIN = 15.0

def _time_left():
    global _START_TIME
    if _START_TIME is None:
        return TIME_LIMIT - SAFETY_MARGIN
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)

class _EnsembleMember:
    def __init__(self, name, predict_fn, score):
        self.name = name
        self.predict_fn = predict_fn
        self.score = score  # Lower is better (MSE)

def _fit_fourier(X, y, max_freq=20, n_harmonics=None):
    """
    Fit a Fourier series.
    """
    x = X[:, 0]
    if n_harmonics is None:
        n_harmonics = max_freq
    # Build design matrix: 1, cos(k*x), sin(k*x) for k=1..n
    cols = [np.ones_like(x)]
    for k in range(1, n_harmonics + 1):
        cols.append(np.cos(k * x))
        cols.append(np.sin(k * x))
    Xf = np.column_stack(cols)
    w, *_ = np.linalg.lstsq(Xf, y, rcond=None)
    
    def predict(X_in):
        xi = X_in[:, 0]
        cols_in = [np.ones_like(xi)]
        for k in range(1, n_harmonics + 1):
            cols_in.append(np.cos(k * xi))
            cols_in.append(np.sin(k * xi))
        Xf_in = np.column_stack(cols_in)
        return Xf_in @ w
    return predict

def _fit_chebyshev(X, y, degree):
    """
    Fit a Chebyshev polynomial.
    """
    x = X[:, 0]
    # Normalize x to [-1, 1] for Chebyshev stability
    x_min, x_max = x.min(), x.max()
    x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
    
    # Build design matrix using Chebyshev polynomials
    # T_0 = 1, T_1 = x, T_{n} = 2x*T_{n-1} - T_{n-2}
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

def _fit_rbf(X, y, gamma):
    """
    Fit an RBF (Gaussian) basis function model.
    Uses all training points as centers.
    """
    x = X[:, 0]
    # Centers are the training points
    centers = x
    # K matrix
    dists = np.abs(x[:, None] - centers[None, :])
    K = np.exp(-gamma * dists**2)
    
    # Add small ridge for stability
    K += 1e-8 * np.eye(len(x))
    
    w, *_ = np.linalg.lstsq(K, y, rcond=None)
    
    def predict(X_in):
        xi = X_in[:, 0]
        dists_in = np.abs(xi[:, None] - centers[None, :])
        K_in = np.exp(-gamma * dists_in**2)
        return K_in @ w
    return predict

def _fit_spline(X, y, s_factor=0.1):
    """
    Fit a Univariate Spline.
    s_factor controls the smoothing.
    """
    x = X[:, 0]
    y = y
    # Sort by x
    idx = np.argsort(x)
    xs = x[idx]
    ys = y[idx]
    
    try:
        # s = s_factor * sum((ys - mean)**2)
        s = s_factor * np.sum((ys - ys.mean())**2)
        # k=3 for cubic spline
        spl = UnivariateSpline(xs, ys, k=3, s=s)
        def predict(X_in):
            return spl(X_in[:, 0])
        return predict
    except:
        return lambda X_in: np.full(len(X_in), y.mean())

def _cross_validate(X, y, fit_func, n_folds=5, seed=42):
    """
    Perform k-fold cross-validation.
    Returns the mean validation MSE.
    """
    n = len(y)
    indices = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    
    mse_scores = []
    for i in range(n_folds):
        val_idx = folds[i]
        train_idx = np.setdiff1d(indices, val_idx)
        
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]
        
        try:
            predict_fn = fit_func(X_train, y_train)
            y_pred = predict_fn(X_val)
            if not np.all(np.isfinite(y_pred)):
                return np.inf
            mse = np.mean((y_pred - y_val)**2)
            mse_scores.append(mse)
        except:
            return np.inf
            
    return np.mean(mse_scores)

def _fit_ensemble(X, y, time_budget):
    """
    Fit an ensemble of models.
    """
    x = X[:, 0]
    
    # Define grid searches for each model type
    # We will evaluate each model configuration using CV
    
    candidates = []
    
    # 1. Fourier Series
    fourier_harmonics = [5, 10, 15, 20]
    for h in fourier_harmonics:
        if _time_left() < 5: break
        fit_fn = lambda X, y, h=h: _fit_fourier(X, y, n_harmonics=h)
        score = _cross_validate(X, y, fit_fn)
        if np.isfinite(score):
            candidates.append(_EnsembleMember(f"fourier_{h}", fit_fn(X, y), score))
    
    # 2. Chebyshev Polynomials
    cheb_degrees = [3, 5, 7, 9, 11]
    for d in cheb_degrees:
        if _time_left() < 5: break
        fit_fn = lambda X, y, d=d: _fit_chebyshev(X, y, degree=d)
        score = _cross_validate(X, y, fit_fn)
        if np.isfinite(score):
            candidates.append(_EnsembleMember(f"cheb_{d}", fit_fn(X, y), score))
    
    # 3. RBF
    # Gamma is inverse of length scale squared.
    # Range: 0.1 (wide) to 10.0 (narrow)
    rbfs = [0.5, 1.0, 2.0, 5.0, 10.0]
    for g in rbfs:
        if _time_left() < 5: break
        fit_fn = lambda X, y, g=g: _fit_rbf(X, y, gamma=g)
        score = _cross_validate(X, y, fit_fn)
        if np.isfinite(score):
            candidates.append(_EnsembleMember(f"rbf_{g}", fit_fn(X, y), score))
            
    # 4. Spline
    spline_factors = [0.01, 0.05, 0.1, 0.5]
    for sf in spline_factors:
        if _time_left() < 5: break
        fit_fn = lambda X, y, sf=sf: _fit_spline(X, y, s_factor=sf)
        score = _cross_validate(X, y, fit_fn)
        if np.isfinite(score):
            candidates.append(_EnsembleMember(f"spline_{sf}", fit_fn(X, y), score))

    if not candidates:
        # Fallback: simple polynomial
        return _fit_chebyshev(X, y, degree=5)
        
    # Sort by score
    candidates.sort(key=lambda m: m.score)
    
    # Select top K candidates for ensemble
    # Weight by inverse score (or softmax)
    top_k = min(5, len(candidates))
    top_candidates = candidates[:top_k]
    
    # If the best is significantly better than the others, just use the best
    # Otherwise, ensemble them
    best_score = top_candidates[0].score
    second_best_score = top_candidates[1].score if len(top_candidates) > 1 else np.inf
    
    # Heuristic: If the best is much better (e.g., 10x lower MSE), trust it alone.
    # This avoids "diluting" a perfect fit with mediocre ones.
    if best_score < 1e-6 or (second_best_score > 10 * best_score):
        return top_candidates[0].predict_fn
        
    # Otherwise, weighted average
    # Weights = 1 / (score + epsilon)
    eps = 1e-9
    weights = np.array([1.0 / (c.score + eps) for c in top_candidates])
    weights = weights / weights.sum()
    
    def predict(X_in):
        preds = np.column_stack([c.predict_fn(X_in) for c in top_candidates])
        return preds @ weights
        
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
            predict_fn = _fit_ensemble(X, y, _time_left())
            # Check if prediction is valid
            test_pred = predict_fn(X)
            if np.all(np.isfinite(test_pred)):
                return predict_fn
    except Exception as e:
        pass
    
    # Fallback to simple polynomial
    try:
        if _time_left() > 10:
            return _fit_chebyshev(X, y, degree=5)
    except:
        pass
        
    # Final fallback: constant
    y_mean = y.mean()
    return lambda X_in: np.full(len(X_in), y_mean)
