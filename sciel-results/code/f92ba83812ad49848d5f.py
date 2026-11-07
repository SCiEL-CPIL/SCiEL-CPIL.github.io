"""
Symbolic regression over 4 hidden 1-D equations.
Strategy:
1. Fit multiple candidate models: PCHIP, Cubic Spline, GP (Matern 5/2), RBF, Chebyshev.
2. Use cross-validation to select the best model.
3. If GP is available and fast enough, use it as a strong baseline.
"""
import time
import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline, Rbf
from scipy.optimize import minimize

_DEADLINE = time.time() + 285.0  # safety margin before 330s kill


def _time_ok():
    return time.time() < _DEADLINE


def _fit_pchip(X, y):
    """Fit a PCHIP interpolator."""
    X = np.asarray(X, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    order = np.argsort(X)
    Xs = X[order]
    ys = y[order]
    try:
        pchip = PchipInterpolator(Xs, ys)
        def predict_pchip(Xq):
            Xq = np.asarray(Xq, dtype=float)
            flat = Xq.flatten()
            return pchip(flat).reshape(Xq.shape)
        return predict_pchip
    except Exception:
        return lambda Xq: np.interp(np.asarray(Xq).flatten(), Xs, ys).reshape(np.asarray(Xq).shape)


def _fit_spline(X, y):
    """Fit a natural cubic spline."""
    X = np.asarray(X, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    order = np.argsort(X)
    Xs = X[order]
    ys = y[order]
    try:
        cs = CubicSpline(Xs, ys, bc_type='natural')
        def predict_spline(Xq):
            Xq = np.asarray(Xq, dtype=float)
            flat = Xq.flatten()
            return cs(flat).reshape(Xq.shape)
        return predict_spline
    except Exception:
        return _fit_pchip(X, y)


def _matern52_kernel(x1, x2, length_scale, variance):
    """Matern 5/2 kernel."""
    r = np.sqrt(np.sum((x1 - x2) ** 2, axis=-1))
    return variance * (1 + np.sqrt(5) * r / length_scale) * np.exp(-np.sqrt(5) * r / length_scale)


def _fit_gp(X, y, n_iter=50):
    """Fit a Gaussian Process with Matern 5/2 kernel using scipy.optimize."""
    X = np.asarray(X, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    n = len(X)
    
    # Normalize X for better conditioning
    x_min, x_max = X.min(), X.max()
    if x_max - x_min < 1e-12:
        x_max = x_min + 1.0
    X_norm = (X - x_min) / (x_max - x_min)
    
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std < 1e-12:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    # Initial hyperparameters
    length_scale = 0.1
    variance = 1.0
    noise = 1e-6
    
    def neg_log_marginal(params):
        ls, var, noise_val = np.exp(params[0]), np.exp(params[1]), np.exp(params[2])
        K = _matern52_kernel(X_norm.reshape(-1, 1), X_norm.reshape(-1, 1), ls, var)
        K += noise_val * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
            nll = 0.5 * y_norm @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * n * np.log(2 * np.pi)
            return nll
        except np.linalg.LinAlgError:
            return 1e10
    
    # Optimize hyperparameters
    x0 = np.log([length_scale, variance, noise])
    try:
        result = minimize(neg_log_marginal, x0, method='L-BFGS-B', options={'maxiter': n_iter})
        ls, var, noise_val = np.exp(result.x[0]), np.exp(result.x[1]), np.exp(result.x[2])
    except Exception:
        ls, var, noise_val = length_scale, variance, noise
    
    K = _matern52_kernel(X_norm.reshape(-1, 1), X_norm.reshape(-1, 1), ls, var)
    K += noise_val * np.eye(n)
    
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
    except np.linalg.LinAlgError:
        return None
    
    def predict_gp(Xq):
        Xq = np.asarray(Xq, dtype=float)
        flat = Xq.flatten()
        Xq_norm = (flat - x_min) / (x_max - x_min)
        K_s = _matern52_kernel(Xq_norm.reshape(-1, 1), X_norm.reshape(-1, 1), ls, var)
        pred_norm = K_s @ alpha
        return (pred_norm * y_std + y_mean).reshape(Xq.shape)
    
    return predict_gp


def _fit_rbf(X, y, kernel='thin_plate_spline', epsilon=1.0):
    """Fit an RBF interpolator."""
    X = np.asarray(X, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    try:
        rbf = Rbf(X, y, function=kernel, epsilon=epsilon)
        def predict_rbf(Xq):
            Xq = np.asarray(Xq, dtype=float)
            flat = Xq.flatten()
            return rbf(flat).reshape(Xq.shape)
        return predict_rbf
    except Exception:
        return None


def _fit_chebyshev(X, y, degree):
    """Fit a Chebyshev polynomial of given degree."""
    X = np.asarray(X, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    x_min, x_max = X.min(), X.max()
    if x_max - x_min < 1e-12:
        x_max = x_min + 1.0
    X_norm = (X - (x_max + x_min) / 2) / ((x_max - x_min) / 2)
    
    try:
        c = np.polynomial.chebyshev.chebfit(X_norm, y, degree)
        def predict_cheb(Xq):
            Xq = np.asarray(Xq, dtype=float)
            flat = Xq.flatten()
            X_norm_q = (flat - (x_max + x_min) / 2) / ((x_max - x_min) / 2)
            y_pred = np.polynomial.chebyshev.chebval(X_norm_q, c)
            return y_pred.reshape(Xq.shape)
        return predict_cheb
    except Exception:
        return None


def _cross_validate(X, y, fit_func, n_folds=10, seed=42):
    """Simple k-fold CV to estimate normalized MSE."""
    rng = np.random.RandomState(seed)
    n = len(X)
    indices = rng.permutation(n)
    folds = np.array_split(indices, n_folds)
    nmes = []
    for i in range(n_folds):
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
    return np.mean(nmes)


def fit(X_train, y_train):
    """
    Fit a regression model using ensemble of PCHIP, Spline, GP, RBF, and Chebyshev.
    Returns a predict callable.
    """
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    d = X.shape[1]
    X_use = X[:, 0]
    
    t_start = time.time()
    
    # Handle constant y
    y_var = np.var(y)
    if y_var < 1e-12:
        c = np.mean(y)
        def predict_const(Xq):
            Xq = np.asarray(Xq)
            return np.full(len(Xq), c)
        return predict_const
    
    models = []
    cv_scores = []
    
    # 1. PCHIP
    if _time_ok():
        try:
            pchip_pred = _fit_pchip(X_use, y)
            models.append(pchip_pred)
            cv = _cross_validate(X_use, y, _fit_pchip, n_folds=10, seed=42)
            cv_scores.append(cv)
        except Exception:
            pass
    
    # 2. Cubic Spline
    if _time_ok():
        try:
            spline_pred = _fit_spline(X_use, y)
            models.append(spline_pred)
            cv = _cross_validate(X_use, y, _fit_spline, n_folds=10, seed=42)
            cv_scores.append(cv)
        except Exception:
            pass
    
    # 3. GP (Matern 5/2)
    if _time_ok():
        try:
            gp_pred = _fit_gp(X_use, y, n_iter=30)
            if gp_pred:
                models.append(gp_pred)
                cv = _cross_validate(X_use, y, lambda Xs, ys: _fit_gp(Xs, ys, n_iter=20), n_folds=5, seed=42)
                cv_scores.append(cv)
        except Exception:
            pass
    
    # 4. RBF Thin Plate Spline
    if _time_ok():
        try:
            rbf_pred = _fit_rbf(X_use, y, kernel='thin_plate_spline')
            if rbf_pred:
                models.append(rbf_pred)
                cv = _cross_validate(X_use, y, lambda Xs, ys: _fit_rbf(Xs, ys, kernel='thin_plate_spline'), n_folds=10, seed=42)
                cv_scores.append(cv)
        except Exception:
            pass
    
    # 5. RBF Multiquadric
    if _time_ok():
        try:
            rbf_pred = _fit_rbf(X_use, y, kernel='multiquadric', epsilon=1.0)
            if rbf_pred:
                models.append(rbf_pred)
                cv = _cross_validate(X_use, y, lambda Xs, ys: _fit_rbf(Xs, ys, kernel='multiquadric', epsilon=1.0), n_folds=10, seed=42)
                cv_scores.append(cv)
        except Exception:
            pass
    
    # 6. Chebyshev degree 20
    if _time_ok():
        try:
            cheb_pred = _fit_chebyshev(X_use, y, 20)
            if cheb_pred:
                models.append(cheb_pred)
                cv = _cross_validate(X_use, y, lambda Xs, ys: _fit_chebyshev(Xs, ys, 20), n_folds=10, seed=42)
                cv_scores.append(cv)
        except Exception:
            pass
    
    # 7. Chebyshev degree 50
    if _time_ok():
        try:
            cheb_pred = _fit_chebyshev(X_use, y, 50)
            if cheb_pred:
                models.append(cheb_pred)
                cv = _cross_validate(X_use, y, lambda Xs, ys: _fit_chebyshev(Xs, ys, 50), n_folds=10, seed=42)
                cv_scores.append(cv)
        except Exception:
            pass
    
    # If no models succeeded, use linear
    if not models:
        Xs = X_use.reshape(-1, 1)
        w, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(Xs)), Xs]), y, rcond=None)
        def predict_linear(Xq):
            Xq = np.asarray(Xq)
            if Xq.ndim == 1:
                Xq = Xq.reshape(-1, 1)
            else:
                Xq = Xq[:, 0:1]
            return np.column_stack([np.ones(len(Xq)), Xq]) @ w
        return predict_linear
    
    # Pick best model by CV score
    best_idx = np.argmin(cv_scores)
    best_model = models[best_idx]
    
    # Try ensemble of top 2 if close
    if len(models) >= 2:
        sorted_idx = np.argsort(cv_scores)
        if cv_scores[sorted_idx[1]] < cv_scores[sorted_idx[0]] * 1.2:
            m1 = models[sorted_idx[0]]
            m2 = models[sorted_idx[1]]
            def predict_ensemble(Xq):
                p1 = m1(Xq)
                p2 = m2(Xq)
                return 0.5 * p1 + 0.5 * p2
            if cv_scores[sorted_idx[0]] < 0.1 and cv_scores[sorted_idx[1]] < 0.1:
                best_model = predict_ensemble
    
    def predict(Xq):
        Xq = np.asarray(Xq, dtype=float)
        if Xq.ndim == 1:
            Xq = Xq.reshape(-1, 1)
        if Xq.shape[1] > 1:
            Xq = Xq[:, 0:1]
        result = best_model(Xq)
        result = np.asarray(result, dtype=float)
        result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)
        return result
    
    return predict
