import numpy as np
import time
from scipy.interpolate import PchipInterpolator, CubicSpline, Rbf
from scipy.optimize import minimize_scalar

# Global deadline guard (330s kill, keep 15s margin)
_GLOBAL_DEADLINE = time.time() + 315.0

def _time_ok():
    return time.time() < _GLOBAL_DEADLINE

def _fit_pchip(X, y):
    X = np.asarray(X, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    order = np.argsort(X)
    Xs, ys = X[order], y[order]
    if np.any(np.diff(Xs) <= 0):
        # Deduplicate
        unique_X, idx = np.unique(Xs, return_index=True)
        ys = ys[np.sort(idx)]
        Xs = unique_X
    try:
        pchip = PchipInterpolator(Xs, ys)
        def predict(Xq):
            Xq = np.asarray(Xq, dtype=float)
            flat = Xq.flatten()
            return pchip(flat).reshape(Xq.shape)
        return predict
    except Exception:
        def predict_interp(Xq):
            Xq = np.asarray(Xq, dtype=float)
            flat = Xq.flatten()
            return np.interp(flat, Xs, ys).reshape(Xq.shape)
        return predict_interp

def _fit_spline(X, y):
    X = np.asarray(X, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    order = np.argsort(X)
    Xs, ys = X[order], y[order]
    if np.any(np.diff(Xs) <= 0):
        unique_X, idx = np.unique(Xs, return_index=True)
        ys = ys[np.sort(idx)]
        Xs = unique_X
    try:
        cs = CubicSpline(Xs, ys, bc_type='natural')
        def predict(Xq):
            Xq = np.asarray(Xq, dtype=float)
            flat = Xq.flatten()
            return cs(flat).reshape(Xq.shape)
        return predict
    except Exception:
        return _fit_pchip(X, y)

def _fit_rbf_scipy(X, y, func='thin_plate_spline', epsilon=1.0):
    X = np.asarray(X, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    try:
        rbf = Rbf(X, y, function=func, epsilon=epsilon)
        def predict(Xq):
            Xq = np.asarray(Xq, dtype=float)
            flat = Xq.flatten()
            return rbf(flat).reshape(Xq.shape)
        return predict
    except Exception:
        return None

def _fit_chebyshev(X, y, degree):
    X = np.asarray(X, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    x_min, x_max = X.min(), X.max()
    if x_max - x_min < 1e-12:
        x_max = x_min + 1.0
    X_norm = (X - (x_max + x_min) / 2) / ((x_max - x_min) / 2)
    try:
        c = np.polynomial.chebyshev.chebfit(X_norm, y, degree)
        def predict(Xq):
            Xq = np.asarray(Xq, dtype=float)
            flat = Xq.flatten()
            X_norm_q = (flat - (x_max + x_min) / 2) / ((x_max - x_min) / 2)
            y_pred = np.polynomial.chebyshev.chebval(X_norm_q, c)
            return y_pred.reshape(Xq.shape)
        return predict
    except Exception:
        return None

def _fit_gp_matern(X, y, length_scale, noise=1e-10):
    """Gaussian Process with Matern 5/2 kernel, exact interpolation for noiseless data."""
    X = np.asarray(X, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    n = len(X)
    
    # Matern 5/2 kernel
    def matern52(X1, X2, l):
        diff = X1[:, None] - X2[None, :]
        r = np.abs(diff) / l
        sqrt3_r = np.sqrt(3.0) * r
        K = (1.0 + sqrt3_r + r**2 / 3.0) * np.exp(-sqrt3_r)
        return K
    
    try:
        K = matern52(X, X, length_scale)
        K += noise * np.eye(n)
        
        # Solve K w = y
        L = np.linalg.cholesky(K)
        w = np.linalg.solve(L.T, np.linalg.solve(L, y))
        
        def predict(Xq):
            Xq = np.asarray(Xq, dtype=float).reshape(-1)
            K_s = matern52(Xq, X, length_scale)
            y_pred = K_s.dot(w)
            return y_pred.reshape(np.asarray(Xq).shape)
        
        return predict
    except Exception:
        return None

def _cross_validate(X, y, fit_func, n_folds=10, seed=42):
    """k-fold CV returning mean normalized MSE."""
    rng = np.random.RandomState(seed)
    n = len(X)
    indices = rng.permutation(n)
    folds = np.array_split(indices, n_folds)
    nmes = []
    for i in range(n_folds):
        if not _time_ok():
            break
        val_idx = folds[i]
        trn_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        try:
            pred_func = fit_func(X[trn_idx], y[trn_idx])
            if pred_func is None:
                nmes.append(1e10)
                continue
            y_pred = pred_func(X[val_idx])
            y_true = y[val_idx]
            var_y = np.var(y_true)
            if var_y < 1e-12:
                nmes.append(0.0)
            else:
                nmes.append(np.mean((y_pred - y_true) ** 2) / var_y)
        except Exception:
            nmes.append(1e10)
    if not nmes:
        return 1e10
    return np.mean(nmes)

def fit(X_train, y_train):
    t_start = time.time()
    
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X_use = X[:, 0].copy()
    y = y.copy()
    
    # Handle constant y
    y_var = np.var(y)
    if y_var < 1e-12:
        c = np.mean(y)
        def predict_const(Xq):
            Xq = np.asarray(Xq)
            return np.full(len(Xq), c)
        return predict_const
    
    # Deduplicate X values
    unique_X, inverse, counts = np.unique(X_use, return_inverse=True, return_counts=True)
    if len(unique_X) != len(X_use):
        y_sum = np.zeros_like(unique_X, dtype=float)
        np.add.at(y_sum, inverse, y)
        y_use = y_sum / counts
        X_use = unique_X
    else:
        y_use = y
    
    n = len(X_use)
    if n < 2:
        c = np.mean(y_use)
        def predict_const(Xq):
            Xq = np.asarray(Xq)
            return np.full(len(Xq), c)
        return predict_const
    
    models = []
    cv_scores = []
    
    # 1. PCHIP
    if _time_ok():
        try:
            p = _fit_pchip(X_use, y_use)
            models.append(p)
            cv = _cross_validate(X_use, y_use, _fit_pchip, n_folds=10)
            cv_scores.append(cv)
        except Exception:
            pass
    
    # 2. Cubic Spline
    if _time_ok():
        try:
            p = _fit_spline(X_use, y_use)
            models.append(p)
            cv = _cross_validate(X_use, y_use, _fit_spline, n_folds=10)
            cv_scores.append(cv)
        except Exception:
            pass
    
    # 3. RBF Thin Plate Spline
    if _time_ok():
        try:
            p = _fit_rbf_scipy(X_use, y_use, 'thin_plate_spline')
            if p:
                models.append(p)
                cv = _cross_validate(X_use, y_use, lambda Xs, ys: _fit_rbf_scipy(Xs, ys, 'thin_plate_spline'), n_folds=10)
                cv_scores.append(cv)
        except Exception:
            pass
    
    # 4. RBF Multiquadric
    if _time_ok():
        try:
            p = _fit_rbf_scipy(X_use, y_use, 'multiquadric', epsilon=1.0)
            if p:
                models.append(p)
                cv = _cross_validate(X_use, y_use, lambda Xs, ys: _fit_rbf_scipy(Xs, ys, 'multiquadric', 1.0), n_folds=10)
                cv_scores.append(cv)
        except Exception:
            pass
    
    # 5. Chebyshev degree 20
    if _time_ok():
        try:
            p = _fit_chebyshev(X_use, y_use, 20)
            if p:
                models.append(p)
                cv = _cross_validate(X_use, y_use, lambda Xs, ys: _fit_chebyshev(Xs, ys, 20), n_folds=10)
                cv_scores.append(cv)
        except Exception:
            pass
    
    # 6. Chebyshev degree 40
    if _time_ok():
        try:
            p = _fit_chebyshev(X_use, y_use, 40)
            if p:
                models.append(p)
                cv = _cross_validate(X_use, y_use, lambda Xs, ys: _fit_chebyshev(Xs, ys, 40), n_folds=10)
                cv_scores.append(cv)
        except Exception:
            pass
    
    # 7. GP with Matern 5/2 - try several length scales
    if _time_ok():
        x_range = X_use.max() - X_use.min()
        if x_range < 1e-12:
            x_range = 1.0
        # Try length scales relative to data range
        for ls_frac in [0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0]:
            if not _time_ok():
                break
            ls = ls_frac * x_range
            try:
                p = _fit_gp_matern(X_use, y_use, length_scale=ls)
                if p:
                    models.append(p)
                    cv = _cross_validate(X_use, y_use, lambda Xs, ys: _fit_gp_matern(Xs, ys, length_scale=ls), n_folds=10)
                    cv_scores.append(cv)
            except Exception:
                pass
    
    # Fallback: linear
    if not models:
        Xs = X_use.reshape(-1, 1)
        A = np.column_stack([np.ones(len(Xs)), Xs])
        w, *_ = np.linalg.lstsq(A, y_use, rcond=None)
        def predict_linear(Xq):
            Xq = np.asarray(Xq, dtype=float)
            if Xq.ndim == 1:
                Xq = Xq.reshape(-1, 1)
            else:
                Xq = Xq[:, 0:1]
            return np.column_stack([np.ones(len(Xq)), Xq]) @ w
        return predict_linear
    
    # Pick best model
    best_idx = np.argmin(cv_scores)
    best_model = models[best_idx]
    best_score = cv_scores[best_idx]
    
    # Try ensemble of top 2 if close
    if len(models) >= 2:
        sorted_idx = np.argsort(cv_scores)
        if cv_scores[sorted_idx[1]] < cv_scores[sorted_idx[0]] * 1.5:
            m1 = models[sorted_idx[0]]
            m2 = models[sorted_idx[1]]
            def predict_ensemble(Xq):
                p1 = m1(Xq)
                p2 = m2(Xq)
                return 0.5 * p1 + 0.5 * p2
            # Quick validation
            try:
                test_pred = predict_ensemble(X_use[:10])
                if np.all(np.isfinite(test_pred)):
                    best_model = predict_ensemble
            except Exception:
                pass
    
    return best_model
