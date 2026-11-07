"""Symbolic regression over 4 hidden 1-D noiseless equations.

Structural change: replace the ad-hoc basis-family soup with a principled
Gaussian Process regression whose kernel is a sum of Automatic Relevance
Determination (ARD) RBF + constant + white-noise, with hyper-parameters
(length_scale, signal_var, noise_var) learned by maximizing the log marginal
likelihood via L-BFGS-B.  For noiseless 1-D data the GP posterior is
essentially an interpolator whose smoothness is dictated by the learned
length scale, which captures derivative structure of the hidden equation
rather than committing to a fixed basis.  A small ensemble of length scales
is kept as a safety net, and a hold-out validation split selects the best
predictor.  A final fallback to a polynomial / spline is retained in case
the GP path fails entirely.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.interpolate import PchipInterpolator, CubicSpline, UnivariateSpline
import warnings

warnings.filterwarnings('ignore')

_START = time.time()
_DEADLINE = _START + 285.0  # safety margin before 330s kill


def _now():
    return time.time()


def _remaining():
    return _DEADLINE - _now()


# ---------------------------------------------------------------------------
# GP machinery
# ---------------------------------------------------------------------------
def _gp_log_mll(theta, x, y, x_mean, x_std, y_mean, y_std):
    """Log marginal likelihood of a GP with ARD RBF + const + noise kernel.

    theta = [log_length_scale, log_signal_var, log_noise_var]
    """
    try:
        ls = float(np.exp(theta[0]))
        sig2 = float(np.exp(theta[1]))
        ns2 = float(np.exp(theta[2]))
    except Exception:
        return -1e12

    if not np.isfinite(ls) or ls <= 0 or not np.isfinite(sig2) or sig2 <= 0 \
            or not np.isfinite(ns2) or ns2 <= 0:
        return -1e12

    n = x.size
    xn = (x - x_mean) / x_std
    diff = xn[:, None] - xn[None, :]
    K = sig2 * np.exp(-0.5 * diff * diff / (ls * ls)) + ns2 * np.eye(n)

    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        # Add jitter and retry once
        try:
            L = np.linalg.cholesky(K + 1e-8 * np.eye(n))
        except np.linalg.LinAlgError:
            return -1e12

    # Solve L a = y
    a = np.linalg.solve(L, y - y_mean)
    # log det K = 2 * sum log diag(L)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    mll = -0.5 * (n * np.log(2.0 * np.pi) + logdet + float(a @ a))
    if not np.isfinite(mll):
        return -1e12
    return mll


def _fit_gp(X_train, y_train, init_ls_frac=0.3, max_iters=200):
    """Fit a GP with ARD RBF + const + noise, learning hyper-parameters."""
    x = X_train[:, 0].copy()
    y = y_train.copy()
    n = x.size
    if n < 3:
        return None

    x_mean = float(x.mean())
    x_std = float(x.std())
    if x_std < 1e-12:
        x_std = 1.0
    y_mean = float(y.mean())
    y_std = float(y.std())
    if y_std < 1e-12:
        y_std = 1.0

    # Initial guess
    ls0 = init_ls_frac  # in normalized units
    sig2_0 = y_std * y_std
    ns2_0 = max(1e-8, (y_std * 1e-3) ** 2)
    theta0 = np.array([np.log(ls0), np.log(sig2_0), np.log(ns2_0)])

    best_theta = theta0.copy()
    best_mll = _gp_log_mll(theta0, x, y, x_mean, x_std, y_mean, y_std)
    if not np.isfinite(best_mll):
        best_mll = -1e12

    try:
        res = minimize(
            _gp_log_mll, theta0,
            args=(x, y, x_mean, x_std, y_mean, y_std),
            method='L-BFGS-B',
            bounds=[(-8.0, 8.0), (-8.0, 8.0), (-20.0, 2.0)],
            options={'maxiter': max_iters, 'ftol': 1e-10, 'gtol': 1e-8},
        )
        if res.success or res.fun < best_mll:
            best_theta = res.x
            best_mll = float(res.fun)
    except Exception:
        pass

    if not np.isfinite(best_mll) and best_mll < -1e10:
        # Optimization failed badly; still try to use the initial guess
        pass

    ls = float(np.exp(best_theta[0]))
    sig2 = float(np.exp(best_theta[1]))
    ns2 = float(np.exp(best_theta[2]))
    if not np.isfinite(ls) or ls <= 0 or not np.isfinite(sig2) or sig2 <= 0 \
            or not np.isfinite(ns2) or ns2 <= 0:
        return None

    xn = (x - x_mean) / x_std
    diff = xn[:, None] - xn[None, :]
    K = sig2 * np.exp(-0.5 * diff * diff / (ls * ls)) + ns2 * np.eye(n)
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y - y_mean))
    except np.linalg.LinAlgError:
        try:
            alpha = np.linalg.solve(K, y - y_mean)
        except np.linalg.LinAlgError:
            return None

    def predict(X):
        xt = np.asarray(X, dtype=float)[:, 0]
        xtn = (xt - x_mean) / x_std
        Ks = sig2 * np.exp(-0.5 * (xtn[:, None] - xn[None, :]) ** 2 / (ls * ls))
        mu = Ks @ alpha + y_mean
        return mu

    return predict


# ---------------------------------------------------------------------------
# Fallback / auxiliary models (kept lean)
# ---------------------------------------------------------------------------
def _fit_spline(X_train, y_train, method='cubic'):
    x = X_train[:, 0]
    y = y_train
    sidx = np.argsort(x)
    xs = x[sidx]
    ys = y[sidx]
    keep = np.concatenate(([True], np.diff(xs) > 1e-12))
    xs, ys = xs[keep], ys[keep]
    if xs.size < 2:
        return None
    if method == 'pchip':
        spl = PchipInterpolator(xs, ys)
    else:
        spl = CubicSpline(xs, ys)

    def predict(X):
        return spl(X[:, 0])

    return predict


def _fit_uni_spline(X_train, y_train, s=0.0, k=3):
    x = X_train[:, 0]
    y = y_train
    sidx = np.argsort(x)
    xs = x[sidx]
    ys = y[sidx]
    keep = np.concatenate(([True], np.diff(xs) > 1e-12))
    xs, ys = xs[keep], ys[keep]
    if xs.size < k + 1:
        return None
    try:
        spl = UnivariateSpline(xs, ys, k=k, s=s)
    except Exception:
        return None

    def predict(X):
        return spl(X[:, 0])

    return predict


def _fit_poly(X_train, y_train, degree):
    x = X_train[:, 0]
    y = y_train
    try:
        coeffs = np.polyfit(x, y, degree)
    except Exception:
        return None

    def predict(X):
        return np.polyval(coeffs, X[:, 0])

    return predict


# ---------------------------------------------------------------------------
# Main fit
# ---------------------------------------------------------------------------
def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    n = len(X_train)
    x = X_train[:, 0]
    y = y_train

    # Hold-out validation split
    rng = np.random.RandomState(42)
    val_n = max(20, int(n * 0.15))
    perm = rng.permutation(n)
    val_idx = perm[:val_n]
    tr_idx = perm[val_n:]
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]

    candidates = []  # (val_mse, predict)

    def _add(pred):
        try:
            yp = np.asarray(pred(X_val), dtype=float)
            if yp.shape != y_val.shape:
                yp = yp.reshape(y_val.shape)
            if not np.all(np.isfinite(yp)):
                return
            mse = float(np.mean((yp - y_val) ** 2))
            if np.isfinite(mse):
                candidates.append((mse, pred))
        except Exception:
            pass

    # ---- GP models with different initial length scales ----
    if _remaining() > 10:
        for frac in (0.1, 0.2, 0.3, 0.5, 0.8, 1.2):
            if _remaining() < 8:
                break
            try:
                pred = _fit_gp(X_tr, y_tr, init_ls_frac=frac)
                if pred is not None:
                    _add(pred)
            except Exception:
                pass

    # ---- GP on full data with a few length scales (no validation split) ----
    if _remaining() > 10:
        for frac in (0.2, 0.5, 1.0):
            if _remaining() < 8:
                break
            try:
                pred = _fit_gp(X_train, y, init_ls_frac=frac)
                if pred is not None:
                    _add(pred)
            except Exception:
                pass

    # ---- Spline fallbacks ----
    if _remaining() > 5:
        for method in ('cubic', 'pchip'):
            try:
                pred = _fit_spline(X_train, y, method)
                if pred is not None:
                    _add(pred)
            except Exception:
                pass

    # ---- UnivariateSpline with small smoothing ----
    if _remaining() > 5:
        for s in (0.0, 1e-6, 1e-4):
            try:
                pred = _fit_uni_spline(X_train, y, s=s, k=3)
                if pred is not None:
                    _add(pred)
            except Exception:
                pass

    # ---- Polynomial fallbacks (low degree only) ----
    if _remaining() > 5:
        for deg in (2, 3, 4, 5, 6, 8, 10):
            if _remaining() < 4:
                break
            try:
                pred = _fit_poly(X_train, y, deg)
                if pred is not None:
                    _add(pred)
            except Exception:
                pass

    # ---- Selection ----
    if not candidates:
        mean_val = float(np.mean(y))
        def predict(X):
            return np.full(X.shape[0], mean_val)
        return predict

    candidates.sort(key=lambda c: c[0])
    best_predict = candidates[0][1]

    def final_predict(X):
        try:
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            yout = np.asarray(best_predict(X), dtype=float)
            yout = np.where(np.isfinite(yout), yout, 0.0)
            return yout
        except Exception:
            return np.zeros(X.shape[0])

    return final_predict
