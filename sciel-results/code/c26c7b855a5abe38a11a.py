"""Symbolic regression: LOESS + global basis + model selection with LOO-CV bandwidth."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar
import warnings

warnings.filterwarnings('ignore')

_TOTAL_BUDGET = 310.0


def _elapsed():
    return time.time() - _GLOBAL_START


def _remaining():
    return _TOTAL_BUDGET - _elapsed()


_GLOBAL_START = time.time()


def _loess_predict(x_new, x, y, bandwidth, degree=2):
    """LOESS prediction at x_new points using local weighted polynomial."""
    x_new = np.asarray(x_new, dtype=np.float64)
    n = len(x)
    if n == 0:
        return np.zeros(len(x_new))
    # bandwidth: fraction of points to use in each local fit
    k = max(3, int(bandwidth * n))
    k = min(k, n)
    preds = np.zeros(len(x_new))
    x_range = x.max() - x.min()
    if x_range < 1e-12:
        return np.full(len(x_new), y.mean())
    for i, xi in enumerate(x_new):
        dists = np.abs(x - xi)
        idx = np.argsort(dists)[:k]
        xi_k = x[idx]
        yi_k = y[idx]
        max_d = dists[idx].max()
        if max_d < 1e-12:
            # All points at same location
            preds[i] = np.mean(yi_k)
            continue
        # Tricube weights
        u = dists[idx] / max_d
        w = (1.0 - u**3)**3
        # Solve weighted least squares
        if degree == 1:
            A = np.column_stack([np.ones(k), xi_k - xi])
        else:
            A = np.column_stack([np.ones(k), xi_k - xi, (xi_k - xi)**2])
        W = np.diag(w)
        try:
            # Weighted least squares: (A^T W A) beta = A^T W y
            ATA = A.T @ (w[:, None] * A)
            ATy = A.T @ (w * yi_k)
            # Regularize
            ATA += 1e-10 * np.eye(ATA.shape[0])
            beta = np.linalg.solve(ATA, ATy)
            preds[i] = beta[0]
        except np.linalg.LinAlgError:
            # Fallback: weighted mean
            preds[i] = np.sum(w * yi_k) / np.sum(w)
    return preds


def _loess_loo_cv(x, y, bandwidth, degree=2):
    """Leave-one-out cross-validation error for LOESS."""
    n = len(x)
    if n < 4:
        return np.inf
    k = max(3, int(bandwidth * n))
    k = min(k, n)
    x_range = x.max() - x.min()
    if x_range < 1e-12:
        return np.var(y)
    errs = []
    for i in range(n):
        dists = np.abs(x - x[i])
        # Exclude point i
        order = np.argsort(dists)
        # Find k nearest excluding i
        idx = [j for j in order if j != i][:k]
        if len(idx) < 3:
            errs.append((y[i] - np.mean(y))**2)
            continue
        xi_k = x[idx]
        yi_k = y[idx]
        max_d = np.max(np.abs(xi_k - x[i]))
        if max_d < 1e-12:
            errs.append((y[i] - np.mean(yi_k))**2)
            continue
        u = np.abs(xi_k - x[i]) / max_d
        w = (1.0 - u**3)**3
        if degree == 1:
            A = np.column_stack([np.ones(len(idx)), xi_k - x[i]])
        else:
            A = np.column_stack([np.ones(len(idx)), xi_k - x[i], (xi_k - x[i])**2])
        ATA = A.T @ (w[:, None] * A)
        ATy = A.T @ (w * yi_k)
        ATA += 1e-10 * np.eye(ATA.shape[0])
        try:
            beta = np.linalg.solve(ATA, ATy)
            pred_i = beta[0]
        except np.linalg.LinAlgError:
            pred_i = np.sum(w * yi_k) / np.sum(w)
        errs.append((y[i] - pred_i)**2)
    return float(np.mean(errs))


def _fit_loess(X, y, bandwidth=None, degree=2):
    """Fit LOESS model; if bandwidth is None, select via LOO-CV."""
    x = X[:, 0].astype(np.float64)
    y = y.astype(np.float64)
    n = len(x)
    if n < 4:
        # Fallback
        c = np.mean(y)
        def predict(X_new):
            return np.full(len(X_new), c)
        return predict

    if bandwidth is None:
        # Search over bandwidths
        best_bw = 0.5
        best_err = np.inf
        for bw in [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            if _remaining() < 5:
                break
            try:
                err = _loess_loo_cv(x, y, bw, degree)
                if err < best_err:
                    best_err = err
                    best_bw = bw
            except Exception:
                pass
        bandwidth = best_bw

    # Final fit: store data for prediction
    x_sorted = x
    y_sorted = y
    bw = bandwidth
    deg = degree

    def predict(X_new):
        return _loess_predict(X_new[:, 0], x_sorted, y_sorted, bw, deg)

    return predict


def _fit_smooth_spline(X, y, s_factor=1.0):
    """Fit UnivariateSpline with smoothing parameter."""
    x = X[:, 0].astype(np.float64)
    y = y.astype(np.float64)
    n = len(x)
    if n < 4:
        c = np.mean(y)
        def predict(X_new):
            return np.full(len(X_new), c)
        return predict
    try:
        s = s_factor * np.var(y) * n
        sp = UnivariateSpline(x, y, s=s, k=4)
        def predict(X_new):
            return sp(X_new[:, 0])
        return predict
    except Exception:
        # Fallback to linear
        w, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), x]), y, rcond=None)
        def predict(X_new):
            xn = X_new[:, 0]
            return w[0] + w[1] * xn
        return predict


def _fit_poly(X, y, deg):
    x = X[:, 0]
    V = np.vander(x, N=deg + 1, increasing=True)
    w, *_ = np.linalg.lstsq(V, y, rcond=None)
    def predict(X_new):
        xn = X_new[:, 0]
        Vn = np.vander(xn, N=deg + 1, increasing=True)
        return Vn @ w
    return predict


def _fit_fourier(X, y, freqs):
    x = X[:, 0]
    cols = [np.ones_like(x)]
    for f in freqs:
        cols.append(np.sin(f * x))
        cols.append(np.cos(f * x))
    F = np.column_stack(cols)
    w, *_ = np.linalg.lstsq(F, y, rcond=None)
    def predict(X_new):
        xn = X_new[:, 0]
        cols = [np.ones_like(xn)]
        for f in freqs:
            cols.append(np.sin(f * xn))
            cols.append(np.cos(f * xn))
        Fn = np.column_stack(cols)
        return Fn @ w
    return predict


def _fit_rational(X, y):
    x = X[:, 0]
    yv = y
    A = np.column_stack([x, np.ones_like(x), -x * yv])
    p, *_ = np.linalg.lstsq(A, yv, rcond=None)
    a, b, c = p
    def predict(X_new):
        xn = X_new[:, 0]
        num = a * xn + b
        den = c * xn + 1.0
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        return num / den
    return predict


def _fit_exp(X, y):
    x = X[:, 0]
    yv = y
    y_min, y_max = yv.min(), yv.max()
    best_err = np.inf
    best_params = None
    for c in np.linspace(y_min, y_max, 15):
        yr = yv - c
        for b in np.linspace(-1.5, 1.5, 25):
            basis = np.exp(b * x)
            denom = np.dot(basis, basis)
            if denom < 1e-12:
                continue
            a = np.dot(basis, yr) / denom
            pred = a * basis + c
            err = np.mean((pred - yv)**2)
            if err < best_err:
                best_err = err
                best_params = (a, b, c)
    if best_params is None:
        w, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(x), x]), yv, rcond=None)
        a, b, c = w[0], w[1], 0.0
    else:
        a, b, c = best_params
    def predict(X_new):
        xn = X_new[:, 0]
        return a * np.exp(b * xn) + c
    return predict


def _fit_gaussian_process(X, y, length_scale, variance):
    x = X[:, 0]
    n = len(x)
    def kernel(a, b):
        d = (a[:, None] - b[None, :])**2
        return variance * np.exp(-0.5 * d / (length_scale**2))
    K = kernel(x, x) + 1e-8 * np.eye(n)
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    except np.linalg.LinAlgError:
        return None
    def predict(X_new):
        xn = X_new[:, 0]
        Ks = kernel(xn, x)
        return Ks @ alpha
    return predict


def _cv_error(predict_fn, X, y, k=5):
    """K-fold CV error."""
    n = len(X)
    if n < k + 2:
        k = max(2, n - 2)
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    errs = []
    for f in folds:
        mask = np.ones(n, dtype=bool)
        mask[f] = False
        Xtr, ytr = X[mask], y[mask]
        Xte, yte = X[f], y[f]
        try:
            p = predict_fn(Xtr, ytr)
            p = np.asarray(p, dtype=np.float64).flatten()
            if not np.all(np.isfinite(p)):
                return np.inf
            err = np.mean((p - yte)**2)
            errs.append(err)
        except Exception:
            return np.inf
    return float(np.mean(errs))


def fit(X_train, y_train):
    global _GLOBAL_START
    _GLOBAL_START = time.time()
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n = len(X)

    if n < 4:
        c = np.mean(y) if n > 0 else 0.0
        def predict(X_new):
            return np.full(len(X_new), c)
        return predict

    # Build candidate models
    candidates = []

    # 1. LOESS with auto bandwidth (primary method)
    try:
        loess_pf = _fit_loess(X, y)
        cv = _cv_error(lambda Xt, yt: _fit_loess(Xt, yt), X, y, k=4)
        candidates.append(('loess_auto', cv, loess_pf))
    except Exception:
        pass

    # 2. LOESS with fixed bandwidths
    for bw in [0.3, 0.5, 0.7]:
        try:
            pf = _fit_loess(X, y, bandwidth=bw)
            cv = _cv_error(lambda Xt, yt, _bw=bw: _fit_loess(Xt, yt, bandwidth=_bw), X, y, k=4)
            candidates.append((f'loess_{bw}', cv, pf))
        except Exception:
            pass

    # 3. Smooth splines
    for sf in [0.01, 0.1, 1.0, 10.0]:
        try:
            pf = _fit_smooth_spline(X, y, s_factor=sf)
            cv = _cv_error(lambda Xt, yt, _sf=sf: _fit_smooth_spline(Xt, yt, s_factor=_sf), X, y, k=4)
            candidates.append((f'spline_{sf}', cv, pf))
        except Exception:
            pass

    # 4. Polynomials
    for deg in [1, 2, 3, 4, 5, 7]:
        try:
            pf = _fit_poly(X, y, deg)
            cv = _cv_error(lambda Xt, yt, _d=deg: _fit_poly(Xt, yt, _d), X, y, k=4)
            candidates.append((f'poly{deg}', cv, pf))
        except Exception:
            pass

    # 5. Fourier
    for fs in [[1.0], [1.0, 2.0], [0.5, 1.0, 1.5], [1.0, 2.0, 3.0]]:
        try:
            pf = _fit_fourier(X, y, fs)
            cv = _cv_error(lambda Xt, yt, _fs=fs: _fit_fourier(Xt, yt, _fs), X, y, k=4)
            candidates.append((f'fourier_{len(fs)}', cv, pf))
        except Exception:
            pass

    # 6. Rational
    try:
        pf = _fit_rational(X, y)
        cv = _cv_error(lambda Xt, yt: _fit_rational(Xt, yt), X, y, k=4)
        candidates.append(('rational', cv, pf))
    except Exception:
        pass

    # 7. Exponential
    try:
        pf = _fit_exp(X, y)
        cv = _cv_error(lambda Xt, yt: _fit_exp(Xt, yt), X, y, k=4)
        candidates.append(('exp', cv, pf))
    except Exception:
        pass

    # 8. GP
    var_y = np.var(y) + 1e-10
    for ls in [0.3, 0.7, 1.5, 3.0]:
        try:
            pf = _fit_gaussian_process(X, y, ls, var_y)
            if pf is None:
                continue
            cv = _cv_error(lambda Xt, yt, _ls=ls: _fit_gaussian_process(Xt, yt, _ls, var_y), X, y, k=3)
            candidates.append((f'gp_{ls}', cv, pf))
        except Exception:
            pass

    # Filter and sort
    candidates = [c for c in candidates if np.isfinite(c[1])]
    candidates.sort(key=lambda t: t[1])

    if not candidates:
        # Fallback
        w, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), X[:, 0]]), y, rcond=None)
        def predict(X_new):
            xn = X_new[:, 0]
            return w[0] + w[1] * xn
        return predict

    # Ensemble top diverse models
    # Pick top candidates, ensuring diversity
    top_models = []
    used_names = set()
    for name, cv, pf in candidates:
        if len(top_models) >= 3:
            break
        # Check diversity: prefer different model families
        base_name = name.split('_')[0] if '_' in name else name
        if base_name not in used_names:
            top_models.append((name, cv, pf))
            used_names.add(base_name)
    # Fill remaining slots with next best
    if len(top_models) < 3:
        for name, cv, pf in candidates:
            if len(top_models) >= 3:
                break
            if (name, cv, pf) not in top_models:
                top_models.append((name, cv, pf))

    if len(top_models) == 1:
        best_pf = top_models[0][2]
        def safe_predict(X_new):
            try:
                r = best_pf(np.asarray(X_new, dtype=np.float64))
                r = np.asarray(r, dtype=np.float64).flatten()
                r = np.nan_to_num(r, nan=0.0, posinf=1e6, neginf=-1e6)
                return r
            except Exception:
                return np.zeros(len(X_new))
        return safe_predict

    # Weighted ensemble
    weights = np.array([1.0 / (cv + 1e-12) for _, cv, _ in top_models])
    weights = weights / weights.sum()

    def safe_predict(X_new):
        try:
            Xn = np.asarray(X_new, dtype=np.float64)
            acc = np.zeros(len(Xn))
            for w, (_, _, pf) in zip(weights, top_models):
                try:
                    r = pf(Xn)
                    r = np.asarray(r, dtype=np.float64).flatten()
                    r = np.nan_to_num(r, nan=0.0, posinf=1e6, neginf=-1e6)
                    acc += w * r
                except Exception:
                    pass
            acc = np.nan_to_num(acc, nan=0.0, posinf=1e6, neginf=-1e6)
            return acc
        except Exception:
            return np.zeros(len(X_new))

    return safe_predict
# EVOLVE-BLOCK-END
