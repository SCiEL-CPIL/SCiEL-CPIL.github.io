"""Symbolic regression: wavelet denoising + global basis + model selection."""
import numpy as np
import time
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar
import warnings

warnings.filterwarnings('ignore')

_TOTAL_BUDGET = 315.0
_GLOBAL_START = time.time()


def _elapsed():
    return time.time() - _GLOBAL_START


def _remaining():
    return _TOTAL_BUDGET - _elapsed()


# --- Wavelet-based denoising (Daubechies-4, soft thresholding) ---

def _d4_filter():
    """Return the Daubechies-4 low-pass and high-pass filter coefficients."""
    # Standard D4 (db4) scaling coefficients (low-pass)
    h = np.array([
        0.2303778133088964, 0.7148465705495647,
        0.6308807679295904, -0.02798376941695368,
        -0.18703481171888114, 0.030841381835560785,
        0.032883011666982345, -0.010597401784997278
    ])
    # High-pass from low-pass (alternating signs, reversed)
    g = np.zeros_like(h)
    for i in range(len(h)):
        g[i] = (-1)**i * h[len(h) - 1 - i]
    return h, g


def _wavelet_denoise(x, y, x_min, x_max, n_levels=4, threshold_factor=1.0):
    """
    Apply wavelet denoising to 1D data.
    1. Interpolate onto a uniform grid of size N (power of 2).
    2. DWT with D4, soft-threshold detail coefficients.
    3. IDWT to get denoised signal.
    4. Return a callable that interpolates the denoised signal at any x.
    """
    n = len(x)
    # Choose grid size: next power of 2 >= n, at least 256
    N = max(256, 1 << int(np.ceil(np.log2(n))))
    # Ensure N is a power of 2
    while N & (N - 1):
        N += 1
    # Limit N to avoid excessive memory
    N = min(N, 4096)

    grid = np.linspace(x_min, x_max, N)
    # Interpolate y onto the grid
    y_grid = np.interp(grid, np.sort(x), y[np.argsort(x)])

    # DWT
    h, g = _d4_filter()
    coeffs = [y_grid.copy()]
    detail_coeffs = []
    sig = y_grid.copy()
    for level in range(n_levels):
        n_sig = len(sig)
        if n_sig < 8:
            break
        # Padding
        pad = 7  # for D4
        sig_padded = np.concatenate([sig[-pad:], sig, sig[:pad]])
        # Convolution
        approx = np.convolve(sig_padded, h, mode='valid')[::2]
        detail = np.convolve(sig_padded, g, mode='valid')[::2]
        coeffs.append(approx)
        detail_coeffs.append(detail)
        sig = approx

    # Soft thresholding
    sigma = np.median(np.abs(detail_coeffs[-1])) / 0.6745 if len(detail_coeffs[-1]) > 0 else 0.0
    threshold = threshold_factor * sigma * np.sqrt(2 * np.log(N))
    for i in range(len(detail_coeffs)):
        detail_coeffs[i] = np.sign(detail_coeffs[i]) * np.maximum(np.abs(detail_coeffs[i]) - threshold, 0)

    # IDWT
    sig = coeffs[0].copy()
    for level in range(len(detail_coeffs) - 1, -1, -1):
        n_sig = len(sig)
        pad = 7
        sig_padded = np.concatenate([sig[-pad:], sig, sig[:pad]])
        # Upsample
        approx_up = np.zeros(2 * n_sig + 2 * pad)
        approx_up[::2] = sig_padded
        detail_up = np.zeros(2 * n_sig + 2 * pad)
        detail_up[::2] = detail_coeffs[level]
        # Convolution with reconstruction filters
        h_rev = h[::-1]
        g_rev = g[::-1]
        sig = np.convolve(approx_up, h_rev, mode='valid') + np.convolve(detail_up, g_rev, mode='valid')
        # Trim to expected length
        expected_len = 2 * n_sig
        sig = sig[:expected_len]
        if expected_len < 8:
            break

    # Trim to grid size
    sig = sig[:N]

    # Interpolate back
    def predict(X_new):
        xn = np.asarray(X_new, dtype=np.float64).flatten()
        return np.interp(xn, grid, sig)

    return predict


def _fit_wavelet(X, y, n_levels=4, threshold_factor=1.0):
    """Fit wavelet-denoised model."""
    x = X[:, 0].astype(np.float64)
    y = y.astype(np.float64)
    n = len(x)
    if n < 8:
        c = np.mean(y)
        def predict(X_new):
            return np.full(len(X_new), c)
        return predict

    x_min, x_max = x.min(), x.max()
    if x_max - x_min < 1e-12:
        c = np.mean(y)
        def predict(X_new):
            return np.full(len(X_new), c)
        return predict

    try:
        return _wavelet_denoise(x, y, x_min, x_max, n_levels, threshold_factor)
    except Exception:
        # Fallback to linear
        w, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), x]), y, rcond=None)
        def predict(X_new):
            xn = X_new[:, 0]
            return w[0] + w[1] * xn
        return predict


# --- LOESS ---

def _loess_predict(x_new, x, y, bandwidth, degree=2):
    """LOESS prediction at x_new points using local weighted polynomial."""
    x_new = np.asarray(x_new, dtype=np.float64)
    n = len(x)
    if n == 0:
        return np.zeros(len(x_new))
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
            preds[i] = np.mean(yi_k)
            continue
        u = dists[idx] / max_d
        w = (1.0 - u**3)**3
        if degree == 1:
            A = np.column_stack([np.ones(k), xi_k - xi])
        else:
            A = np.column_stack([np.ones(k), xi_k - xi, (xi_k - xi)**2])
        try:
            ATA = A.T @ (w[:, None] * A)
            ATy = A.T @ (w * yi_k)
            ATA += 1e-10 * np.eye(ATA.shape[0])
            beta = np.linalg.solve(ATA, ATy)
            preds[i] = beta[0]
        except np.linalg.LinAlgError:
            preds[i] = np.sum(w * yi_k) / np.sum(w)
    return preds


def _fit_loess(X, y, bandwidth=None, degree=2):
    """Fit LOESS model; if bandwidth is None, select via simple heuristic."""
    x = X[:, 0].astype(np.float64)
    y = y.astype(np.float64)
    n = len(x)
    if n < 4:
        c = np.mean(y)
        def predict(X_new):
            return np.full(len(X_new), c)
        return predict

    if bandwidth is None:
        # Use a moderate default
        bandwidth = 0.5

    x_sorted = x
    y_sorted = y
    bw = bandwidth
    deg = degree

    def predict(X_new):
        return _loess_predict(X_new[:, 0], x_sorted, y_sorted, bw, deg)

    return predict


# --- Other models ---

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


def _fit_chebyshev(X, y, deg):
    """Fit Chebyshev polynomial model."""
    x = X[:, 0].astype(np.float64)
    y = y.astype(np.float64)

    x_min = x.min()
    x_max = x.max()
    if x_max - x_min < 1e-10:
        x_max = x_min + 1.0

    x_norm = (2.0 * x - x_min - x_max) / (x_max - x_min)

    T = np.zeros((len(x), deg + 1))
    T[:, 0] = 1.0
    if deg >= 1:
        T[:, 1] = x_norm
        for k in range(2, deg + 1):
            T[:, k] = 2.0 * x_norm * T[:, k - 1] - T[:, k - 2]

    try:
        w = np.linalg.lstsq(T, y, rcond=None)[0]
    except Exception:
        w = np.zeros(deg + 1)

    def predict(X_new):
        xn = X_new[:, 0].astype(np.float64)
        xn_norm = (2.0 * xn - x_min - x_max) / (x_max - x_min)
        Tn = np.zeros((len(xn), deg + 1))
        Tn[:, 0] = 1.0
        if deg >= 1:
            Tn[:, 1] = xn_norm
            for k in range(2, deg + 1):
                Tn[:, k] = 2.0 * xn_norm * Tn[:, k - 1] - Tn[:, k - 2]
        return Tn @ w

    return predict


def _fit_fourier(X, y, n_terms):
    """Fit Fourier series model."""
    x = X[:, 0].astype(np.float64)
    y = y.astype(np.float64)
    n = len(x)

    x_range = x.max() - x.min()
    if x_range < 1e-10:
        x_range = 1.0

    basis = [np.ones(n)]
    for k in range(1, n_terms + 1):
        freq = 2.0 * np.pi * k / x_range
        basis.append(np.sin(freq * x))
        basis.append(np.cos(freq * x))

    F = np.column_stack(basis)

    try:
        w = np.linalg.lstsq(F, y, rcond=None)[0]
    except Exception:
        w = np.zeros(F.shape[1])

    def predict(X_new):
        xn = X_new[:, 0].astype(np.float64)
        basis_new = [np.ones(len(xn))]
        for k in range(1, n_terms + 1):
            freq = 2.0 * np.pi * k / x_range
            basis_new.append(np.sin(freq * xn))
            basis_new.append(np.cos(freq * xn))
        Fn = np.column_stack(basis_new)
        return Fn @ w

    return predict


def _fit_rational(X, y, num_deg=2, den_deg=2):
    """Fit rational function: P(x)/Q(x) where P, Q are polynomials."""
    x = X[:, 0].astype(np.float64)
    y = y.astype(np.float64)
    n = len(x)

    x_mean = x.mean()
    x_std = x.std()
    if x_std < 1e-10:
        x_std = 1.0

    x_scaled = (x - x_mean) / x_std

    n_params = num_deg + den_deg + 1
    A = np.zeros((n, n_params))

    for k in range(num_deg + 1):
        A[:, k] = x_scaled ** k

    for k in range(1, den_deg + 1):
        A[:, num_deg + k] = -y * (x_scaled ** k)

    b = y * np.ones(n)

    try:
        coeff = np.linalg.lstsq(A, b, rcond=None)[0]
        p_coeffs = coeff[:num_deg + 1]
        q_coeffs = coeff[num_deg + 1:]
    except Exception:
        c = np.mean(y)
        def predict(X_new):
            return np.full(len(X_new), c)
        return predict

    def predict(X_new):
        xn = X_new[:, 0].astype(np.float64)
        xn_scaled = (xn - x_mean) / x_std

        P = np.zeros(len(xn))
        for k in range(num_deg + 1):
            P += p_coeffs[k] * (xn_scaled ** k)

        Q = np.ones(len(xn))
        for k in range(1, den_deg + 1):
            Q += q_coeffs[k - 1] * (xn_scaled ** k)

        Q_safe = np.where(np.abs(Q) < 1e-10, 1e-10, Q)
        return P / Q_safe

    return predict


def _fit_exp(X, y):
    """Fit exponential model: a*exp(b*x) + c."""
    x = X[:, 0].astype(np.float64)
    y = y.astype(np.float64)

    x_range = x.max() - x.min()
    if x_range < 1e-10:
        x_range = 1.0

    best_err = np.inf
    best_params = None

    b_values = np.linspace(-3.0 / x_range, 3.0 / x_range, 20)
    for b in b_values:
        exp_x = np.exp(b * x)
        A = np.column_stack([exp_x, np.ones(len(x))])
        try:
            coeff = np.linalg.lstsq(A, y, rcond=None)[0]
            a, c = coeff
            preds = a * exp_x + c
            err = np.mean((preds - y) ** 2)
            if err < best_err:
                best_err = err
                best_params = (a, b, c)
        except Exception:
            pass

    if best_params is None:
        c = np.mean(y)
        def predict(X_new):
            return np.full(len(X_new), c)
        return predict

    a, b, c = best_params

    def predict(X_new):
        xn = X_new[:, 0].astype(np.float64)
        return a * np.exp(b * xn) + c

    return predict


def _fit_rbf(X, y, sigma):
    """Fit Gaussian RBF model."""
    x = X[:, 0].astype(np.float64)
    y = y.astype(np.float64)
    n = len(x)
    reg = 1e-6

    diff = x[:, None] - x[None, :]
    K = np.exp(-diff**2 / (2.0 * sigma**2))
    K_reg = K + reg * np.eye(n)

    try:
        w = np.linalg.solve(K_reg, y)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(K_reg, y, rcond=None)[0]

    def predict(X_new):
        xn = X_new[:, 0].astype(np.float64)
        diff_new = xn[:, None] - x[None, :]
        K_new = np.exp(-diff_new**2 / (2.0 * sigma**2))
        return K_new @ w

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
        if _remaining() < 3:
            return np.inf
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

    # 1. Wavelet denoising (primary new method)
    for nl in [3, 4, 5]:
        for tf in [0.8, 1.0, 1.2, 1.5]:
            if _remaining() < 5:
                break
            try:
                pf = _fit_wavelet(X, y, n_levels=nl, threshold_factor=tf)
                cv = _cv_error(lambda Xt, yt, _nl=nl, _tf=tf: _fit_wavelet(Xt, yt, n_levels=_nl, threshold_factor=_tf), X, y, k=4)
                candidates.append((f'wavelet_{nl}_{tf}', cv, pf))
            except Exception:
                pass

    # 2. LOESS with fixed bandwidths
    for bw in [0.3, 0.5, 0.7]:
        if _remaining() < 5:
            break
        try:
            pf = _fit_loess(X, y, bandwidth=bw)
            cv = _cv_error(lambda Xt, yt, _bw=bw: _fit_loess(Xt, yt, bandwidth=_bw), X, y, k=4)
            candidates.append((f'loess_{bw}', cv, pf))
        except Exception:
            pass

    # 3. Smooth splines
    for sf in [0.01, 0.1, 1.0, 10.0]:
        if _remaining() < 5:
            break
        try:
            pf = _fit_smooth_spline(X, y, s_factor=sf)
            cv = _cv_error(lambda Xt, yt, _sf=sf: _fit_smooth_spline(Xt, yt, s_factor=_sf), X, y, k=4)
            candidates.append((f'spline_{sf}', cv, pf))
        except Exception:
            pass

    # 4. Polynomials
    for deg in [1, 2, 3, 4, 5, 7]:
        if _remaining() < 3:
            break
        try:
            pf = _fit_poly(X, y, deg)
            cv = _cv_error(lambda Xt, yt, _d=deg: _fit_poly(Xt, yt, _d), X, y, k=4)
            candidates.append((f'poly{deg}', cv, pf))
        except Exception:
            pass

    # 5. Chebyshev
    for deg in [2, 4, 6, 8]:
        if _remaining() < 3:
            break
        try:
            pf = _fit_chebyshev(X, y, deg)
            cv = _cv_error(lambda Xt, yt, _d=deg: _fit_chebyshev(Xt, yt, _d), X, y, k=4)
            candidates.append((f'cheb{deg}', cv, pf))
        except Exception:
            pass

    # 6. Fourier
    for nt in [1, 2, 3, 4]:
        if _remaining() < 3:
            break
        try:
            pf = _fit_fourier(X, y, nt)
            cv = _cv_error(lambda Xt, yt, _nt=nt: _fit_fourier(Xt, yt, _nt), X, y, k=4)
            candidates.append((f'fourier_{nt}', cv, pf))
        except Exception:
            pass

    # 7. Rational
    if _remaining() > 3:
        try:
            pf = _fit_rational(X, y)
            cv = _cv_error(lambda Xt, yt: _fit_rational(Xt, yt), X, y, k=4)
            candidates.append(('rational', cv, pf))
        except Exception:
            pass

    # 8. Exponential
    if _remaining() > 3:
        try:
            pf = _fit_exp(X, y)
            cv = _cv_error(lambda Xt, yt: _fit_exp(Xt, yt), X, y, k=4)
            candidates.append(('exp', cv, pf))
        except Exception:
            pass

    # 9. RBF
    x_range = X[:, 0].max() - X[:, 0].min()
    if x_range > 1e-10 and _remaining() > 5:
        for sigma in [0.3 * x_range, 0.7 * x_range, 1.5 * x_range]:
            if _remaining() < 3:
                break
            try:
                pf = _fit_rbf(X, y, sigma)
                cv = _cv_error(lambda Xt, yt, _s=sigma: _fit_rbf(Xt, yt, _s), X, y, k=4)
                candidates.append((f'rbf_{sigma:.2f}', cv, pf))
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
    top_models = []
    used_names = set()
    for name, cv, pf in candidates:
        if len(top_models) >= 3:
            break
        base_name = name.split('_')[0] if '_' in name else name
        if base_name not in used_names:
            top_models.append((name, cv, pf))
            used_names.add(base_name)
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
