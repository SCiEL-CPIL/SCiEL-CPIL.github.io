"""Symbolic regression over 4 hidden equations.
Each equation has 1 input in a known range, 200 noiseless training points,
and 200 test points.  Strategy: use a hold-out validation split to select
among a family of smooth basis expansions (polynomial, Fourier, RBF, spline),
then return the best model.  A deadline guard ensures we finish within 300s.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.interpolate import make_interp_spline, UnivariateSpline
import warnings
warnings.filterwarnings("ignore")

_DEADLINE = time.time() + 290.0  # 40s safety margin under 330s


def _remaining():
    return _DEADLINE - time.time()


def _make_predict(F_train, w, feat_fn):
    """Return a predict callable that is robust to input shape."""
    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        F = feat_fn(X)
        y = F @ w
        y = np.nan_to_num(y, nan=0.0, posinf=1e12, neginf=-1e12)
        return y
    return predict


def _poly_features(X, degree):
    """Polynomial basis up to given degree for a single input."""
    x = X[:, 0]
    cols = [np.ones(len(x))]
    for k in range(1, degree + 1):
        cols.append(x ** k)
    return np.column_stack(cols)


def _fourier_features(X, n_frequencies, x_min, x_max):
    """Fourier basis: sin/cos of integer multiples of fundamental frequency."""
    x = X[:, 0]
    period = x_max - x_min
    cols = [np.ones(len(x))]
    for k in range(1, n_frequencies + 1):
        freq = 2.0 * np.pi * k / period
        cols.append(np.sin(freq * (x - x_min)))
        cols.append(np.cos(freq * (x - x_min)))
    return np.column_stack(cols)


def _rbf_features(X, centers, bandwidth):
    """RBF kernel features with given centers and bandwidth."""
    x = X[:, 0].reshape(-1, 1)
    c = centers.reshape(-1, 1)
    # (n, 1) - (1, k) -> (n, k)
    d2 = (x - c) ** 2
    return np.exp(-d2 / (2.0 * bandwidth ** 2))


def _spline_predict(X_train, y_train, degree=4, s=0.0):
    """Fit a spline and return a predict callable."""
    x = X_train[:, 0]
    # Sort for spline fitting
    idx = np.argsort(x)
    xs = x[idx]
    ys = y_train[idx]
    try:
        spl = UnivariateSpline(xs, ys, k=min(degree, len(xs) - 1), s=s)
    except Exception:
        # Fallback to polynomial if spline fails
        degree = min(4, len(xs) - 1)
        coeffs = np.polyfit(xs, ys, degree)
        def predict_poly(X):
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            return np.polyval(coeffs, X[:, 0])
        return predict_poly

    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        y = spl(X[:, 0])
        y = np.nan_to_num(y, nan=0.0, posinf=1e12, neginf=-1e12)
        return y
    return predict


def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    n = len(X_train)
    x = X_train[:, 0]
    x_min, x_max = x.min(), x.max()
    
    # Hold-out validation split: 80/20
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    val_size = max(20, n // 5)
    val_idx = perm[:val_size]
    tr_idx = perm[val_size:]
    
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    
    candidates = []  # list of (val_mse, predict_fn)
    
    # 1. Polynomial models of various degrees
    for deg in range(1, 13):
        if _remaining() < 5:
            break
        try:
            F_tr = _poly_features(X_tr, deg)
            w, *_ = np.linalg.lstsq(F_tr, y_tr, rcond=None)
            F_val = _poly_features(X_val, deg)
            y_pred_val = F_val @ w
            mse_val = np.mean((y_pred_val - y_val) ** 2)
            var_y = np.var(y_val)
            if var_y > 1e-15:
                nrmse = mse_val / var_y
            else:
                nrmse = mse_val
            feat_fn = lambda X, d=deg: _poly_features(X, d)
            predict_fn = _make_predict(F_tr, w, feat_fn)
            candidates.append((nrmse, predict_fn))
        except Exception:
            continue
    
    # 2. Fourier models with various numbers of harmonics
    for n_harm in range(1, 12):
        if _remaining() < 5:
            break
        try:
            F_tr = _fourier_features(X_tr, n_harm, x_min, x_max)
            w, *_ = np.linalg.lstsq(F_tr, y_tr, rcond=None)
            F_val = _fourier_features(X_val, n_harm, x_min, x_max)
            y_pred_val = F_val @ w
            mse_val = np.mean((y_pred_val - y_val) ** 2)
            var_y = np.var(y_val)
            nrmse = mse_val / var_y if var_y > 1e-15 else mse_val
            feat_fn = lambda X, nh=n_harm, xmin=x_min, xmax=x_max: _fourier_features(X, nh, xmin, xmax)
            predict_fn = _make_predict(F_tr, w, feat_fn)
            candidates.append((nrmse, predict_fn))
        except Exception:
            continue
    
    # 3. RBF models with various numbers of centers and bandwidths
    n_centers_list = [5, 10, 20, 30, 50]
    for nc in n_centers_list:
        if _remaining() < 5:
            break
        # Centers evenly spaced
        centers = np.linspace(x_min, x_max, nc)
        for bw_mult in [0.1, 0.2, 0.5, 1.0]:
            bandwidth = (x_max - x_min) * bw_mult / max(1, nc - 1)
            try:
                F_tr = _rbf_features(X_tr, centers, bandwidth)
                w, *_ = np.linalg.lstsq(F_tr, y_tr, rcond=None)
                F_val = _rbf_features(X_val, centers, bandwidth)
                y_pred_val = F_val @ w
                mse_val = np.mean((y_pred_val - y_val) ** 2)
                var_y = np.var(y_val)
                nrmse = mse_val / var_y if var_y > 1e-15 else mse_val
                feat_fn = lambda X, c=centers, bw=bandwidth: _rbf_features(X, c, bw)
                predict_fn = _make_predict(F_tr, w, feat_fn)
                candidates.append((nrmse, predict_fn))
            except Exception:
                continue
    
    # 4. Spline models with various degrees and smoothing
    for deg in [3, 4, 5, 6]:
        for s in [0.0, 1e-4, 1e-3, 1e-2]:
            if _remaining() < 5:
                break
            try:
                predict_fn = _spline_predict(X_tr, y_tr, degree=deg, s=s)
                y_pred_val = predict_fn(X_val)
                mse_val = np.mean((y_pred_val - y_val) ** 2)
                var_y = np.var(y_val)
                nrmse = mse_val / var_y if var_y > 1e-15 else mse_val
                candidates.append((nrmse, predict_fn))
            except Exception:
                continue
    
    # 5. Mixed basis: polynomial + Fourier combined
    for deg in [1, 2, 3]:
        for n_harm in [2, 4, 6]:
            if _remaining() < 5:
                break
            try:
                F_poly = _poly_features(X_tr, deg)
                F_four = _fourier_features(X_tr, n_harm, x_min, x_max)
                F_tr = np.hstack([F_poly, F_four])
                w, *_ = np.linalg.lstsq(F_tr, y_tr, rcond=None)
                F_val = np.hstack([_poly_features(X_val, deg), _fourier_features(X_val, n_harm, x_min, x_max)])
                y_pred_val = F_val @ w
                mse_val = np.mean((y_pred_val - y_val) ** 2)
                var_y = np.var(y_val)
                nrmse = mse_val / var_y if var_y > 1e-15 else mse_val
                feat_fn = lambda X, d=deg, nh=n_harm, xmin=x_min, xmax=x_max: np.hstack([_poly_features(X, d), _fourier_features(X, nh, xmin, xmax)])
                predict_fn = _make_predict(F_tr, w, feat_fn)
                candidates.append((nrmse, predict_fn))
            except Exception:
                continue
    
    # Select best candidate by validation NRMSE
    if not candidates:
        # Fallback: simple linear fit on all data
        w, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), x]), y_train, rcond=None)
        def predict_linear(X):
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            return w[0] + w[1] * X[:, 0]
        return predict_linear
    
    candidates.sort(key=lambda t: t[0])
    best_nrmse, best_predict = candidates[0]
    
    # If the best model has very high validation error, try fitting on all data with the best basis
    # This can help if the validation split was unlucky
    if best_nrmse > 10.0 and _remaining() > 10:
        # Try refitting the best type on all data
        best_predict_all = None
        for i, (nrmse_i, pred_i) in enumerate(candidates[:5]):
            # We can't easily refit since we don't have the feat_fn stored separately
            # But we can just use the best as is
            break
    
    return best_predict
# EVOLVE-BLOCK-END
