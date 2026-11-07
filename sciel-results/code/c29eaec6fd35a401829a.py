"""
Symbolic regression over 4 hidden 1-D equations.
Strategy:
1. GP with Matern 5/2 kernel (learned length-scale) for smooth generalization.
2. PCHIP (monotone cubic, no overshoot) as a robust baseline.
3. Chebyshev polynomial fits with adaptive degree selection.
4. Spline interpolation.
5. Select the best model via 5-fold cross-validation (normalized MSE).
6. Ensemble of top models if their CV scores are close.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import PchipInterpolator, CubicSpline

_DEADLINE = time.time() + 285.0  # safety margin before 330s kill


def _matern52_kernel(X1, X2, l):
    """Matern 5/2 kernel for 1-D inputs."""
    d = np.abs(X1 - X2) / l
    sqrt5 = np.sqrt(5.0)
    return (1.0 + sqrt5 * d + 5.0 / 3.0 * d * d) * np.exp(-sqrt5 * d)


def _fit_gp(X, y):
    """
    Fit a GP with Matern 5/2 kernel (unit variance, learned length scale).
    Returns predict callable.
    """
    n = len(X)
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    y_orig = y.copy()
    y = y - y.mean()
    y_var = np.var(y_orig)
    if y_var < 1e-12:
        return lambda Xq: np.full(len(Xq), np.mean(y_orig))

    # Standardize X
    x_mean = X.mean()
    x_std = X.std()
    if x_std < 1e-12:
        x_std = 1.0
    Xs = (X - x_mean) / x_std

    jitter = 1e-8

    def neg_log_marginal(theta):
        l = np.exp(theta[0])
        K = _matern52_kernel(Xs, Xs, l)
        K[np.diag_indices_from(K)] += jitter
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10
        alpha = np.linalg.solve(L, y)
        alpha = np.linalg.solve(L.T, alpha)
        nll = 0.5 * y @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * n * np.log(2 * np.pi)
        return nll

    best_theta = np.array([0.0])
    best_nll = neg_log_marginal(best_theta)

    for start in [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]:
        theta0 = np.array([start])
        try:
            res = minimize(neg_log_marginal, theta0, method='Nelder-Mead',
                           options={'maxiter': 30, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < best_nll:
                best_nll = res.fun
                best_theta = res.x
        except Exception:
            pass

    l_opt = np.exp(best_theta[0])
    K = _matern52_kernel(Xs, Xs, l_opt)
    K[np.diag_indices_from(K)] += jitter
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L, y)
        alpha = np.linalg.solve(L.T, alpha)
    except np.linalg.LinAlgError:
        # Fallback: return mean
        return lambda Xq: np.full(len(Xq), np.mean(y_orig))

    def predict_gp(Xq):
        Xq = np.asarray(Xq, dtype=float)
        if Xq.ndim == 1:
            Xq = Xq.reshape(-1, 1)
        Xqs = (Xq - x_mean) / x_std
        Ks = _matern52_kernel(Xqs, Xs, l_opt)
        pred = Ks @ alpha + y.mean()
        return pred

    return predict_gp


def _fit_pchip(X, y):
    """Fit a PCHIP interpolator."""
    X = X.reshape(-1) if X.ndim > 1 else X
    order = np.argsort(X)
    Xs = X[order]
    ys = y[order]
    try:
        pchip = PchipInterpolator(Xs, ys)
    except Exception:
        # Fallback to linear interpolation
        return lambda Xq: np.interp(np.asarray(Xq).flatten(), Xs, ys).reshape(np.asarray(Xq).shape)

    def predict_pchip(Xq):
        Xq = np.asarray(Xq, dtype=float)
        flat = Xq.flatten()
        return pchip(flat).reshape(Xq.shape)

    return predict_pchip


def _fit_chebyshev(X, y, degree):
    """Fit a Chebyshev polynomial of given degree."""
    X = X.reshape(-1) if X.ndim > 1 else X
    x_min, x_max = X.min(), X.max()
    if x_max - x_min < 1e-12:
        x_max = x_min + 1.0
    X_norm = (X - (x_max + x_min) / 2) / ((x_max - x_min) / 2)

    try:
        c = np.polynomial.chebyshev.chebfit(X_norm, y, degree)
    except Exception:
        # Fallback to lower degree
        try:
            c = np.polynomial.chebyshev.chebfit(X_norm, y, 10)
        except Exception:
            return lambda Xq: np.full(len(Xq), np.mean(y))

    def predict_cheb(Xq):
        Xq = np.asarray(Xq, dtype=float)
        flat = Xq.flatten()
        X_norm_q = (flat - (x_max + x_min) / 2) / ((x_max - x_min) / 2)
        y_pred = np.polynomial.chebyshev.chebval(X_norm_q, c)
        return y_pred.reshape(Xq.shape)

    return predict_cheb


def _fit_spline(X, y):
    """Fit a natural cubic spline."""
    X = X.reshape(-1) if X.ndim > 1 else X
    order = np.argsort(X)
    Xs = X[order]
    ys = y[order]
    try:
        cs = CubicSpline(Xs, ys, bc_type='natural')
    except Exception:
        return _fit_pchip(X, y)

    def predict_spline(Xq):
        Xq = np.asarray(Xq, dtype=float)
        flat = Xq.flatten()
        return cs(flat).reshape(Xq.shape)

    return predict_spline


def _cross_validate(X, y, fit_func, n_folds=5, seed=42):
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
            pred = fit_func(X[trn_idx], y[trn_idx])
            y_pred = pred(X[val_idx])
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
    Fit a regression model using ensemble of GP, PCHIP, Chebyshev, and spline.
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

    # 1. PCHIP (fast, monotone, good for non-oscillatory functions)
    if time.time() < _DEADLINE:
        try:
            pchip_pred = _fit_pchip(X_use, y)
            models.append(pchip_pred)
            cv = _cross_validate(X_use, y, _fit_pchip, n_folds=5, seed=42)
            cv_scores.append(cv)
        except Exception:
            pass

    # 2. GP with Matern 5/2
    if time.time() < _DEADLINE:
        try:
            gp_pred = _fit_gp(X_use, y)
            models.append(gp_pred)
            cv = _cross_validate(X_use, y, _fit_gp, n_folds=5, seed=42)
            cv_scores.append(cv)
        except Exception:
            pass

    # 3. Chebyshev degree 10
    if time.time() < _DEADLINE:
        try:
            cheb10_pred = _fit_chebyshev(X_use, y, 10)
            models.append(cheb10_pred)
            cv = _cross_validate(X_use, y, lambda Xs, ys: _fit_chebyshev(Xs, ys, 10), n_folds=5, seed=42)
            cv_scores.append(cv)
        except Exception:
            pass

    # 4. Chebyshev degree 20
    if time.time() < _DEADLINE:
        try:
            cheb20_pred = _fit_chebyshev(X_use, y, 20)
            models.append(cheb20_pred)
            cv = _cross_validate(X_use, y, lambda Xs, ys: _fit_chebyshev(Xs, ys, 20), n_folds=5, seed=42)
            cv_scores.append(cv)
        except Exception:
            pass

    # 5. Chebyshev degree 40
    if time.time() < _DEADLINE:
        try:
            cheb40_pred = _fit_chebyshev(X_use, y, 40)
            models.append(cheb40_pred)
            cv = _cross_validate(X_use, y, lambda Xs, ys: _fit_chebyshev(Xs, ys, 40), n_folds=5, seed=42)
            cv_scores.append(cv)
        except Exception:
            pass

    # 6. Natural cubic spline
    if time.time() < _DEADLINE:
        try:
            spline_pred = _fit_spline(X_use, y)
            models.append(spline_pred)
            cv = _cross_validate(X_use, y, _fit_spline, n_folds=5, seed=42)
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
        if cv_scores[sorted_idx[1]] < cv_scores[sorted_idx[0]] * 1.3:
            m1 = models[sorted_idx[0]]
            m2 = models[sorted_idx[1]]
            def predict_ensemble(Xq):
                p1 = m1(Xq)
                p2 = m2(Xq)
                return 0.5 * p1 + 0.5 * p2
            if cv_scores[sorted_idx[0]] < 0.3 and cv_scores[sorted_idx[1]] < 0.3:
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
