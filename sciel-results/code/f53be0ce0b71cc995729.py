"""Symbolic regression: fast RBF + spline + polynomial ensemble with efficient CV."""
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

_TOTAL_BUDGET = 290.0
_GLOBAL_START = time.time()


def _remaining():
    return _TOTAL_BUDGET - (time.time() - _GLOBAL_START)


def _fit_rbf_fast(X, y, sigma):
    """Fit Gaussian RBF model with precomputed kernel."""
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


def _fit_poly_fast(X, y, deg):
    """Fit polynomial model with scaled input."""
    x = X[:, 0].astype(np.float64)
    y = y.astype(np.float64)

    x_mean = x.mean()
    x_std = x.std()
    if x_std < 1e-10:
        x_std = 1.0

    x_scaled = (x - x_mean) / x_std
    V = np.vander(x_scaled, N=deg + 1, increasing=True)

    try:
        w = np.linalg.lstsq(V, y, rcond=None)[0]
    except Exception:
        w = np.zeros(deg + 1)

    def predict(X_new):
        xn = X_new[:, 0].astype(np.float64)
        xn_scaled = (xn - x_mean) / x_std
        Vn = np.vander(xn_scaled, N=deg + 1, increasing=True)
        return Vn @ w

    return predict


def _fit_spline_fast(X, y, s_factor):
    """Fit B-spline model."""
    from scipy.interpolate import UnivariateSpline

    x = X[:, 0].astype(np.float64)
    y = y.astype(np.float64)
    n = len(x)

    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    s = max(s_factor * np.var(y) * n, 1e-10)

    try:
        sp = UnivariateSpline(x_sorted, y_sorted, s=s, k=4)

        def predict(X_new):
            return sp(X_new[:, 0].astype(np.float64))

        return predict
    except Exception:
        A = np.column_stack([np.ones(n), X[:, 0]])
        try:
            w = np.linalg.lstsq(A, y, rcond=None)[0]

            def predict(X_new):
                An = np.column_stack([np.ones(len(X_new)), X_new[:, 0].astype(np.float64)])
                return An @ w

            return predict
        except Exception:
            c = np.mean(y)

            def predict(X_new):
                return np.full(len(X_new), c)

            return predict


def _fit_fourier_fast(X, y, n_terms):
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


def _fit_chebyshev(X, y, deg):
    """Fit Chebyshev polynomial model."""
    x = X[:, 0].astype(np.float64)
    y = y.astype(np.float64)

    x_min = x.min()
    x_max = x.max()
    if x_max - x_min < 1e-10:
        x_max = x_min + 1.0

    x_norm = (2.0 * x - x_min - x_max) / (x_max - x_min)

    # Build Chebyshev basis
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


def _fit_loess(X, y, bandwidth, deg=2):
    """Fit LOESS (local polynomial regression) model."""
    x = X[:, 0].astype(np.float64)
    y = y.astype(np.float64)
    n = len(x)

    # Precompute for each training point the local fit
    # For prediction, we'll do local regression at each test point
    # This is O(n^2 * deg) but n=200 so it's manageable

    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    def predict(X_new):
        xn = X_new[:, 0].astype(np.float64)
        preds = np.zeros(len(xn))
        for i, xi in enumerate(xn):
            # Compute distances
            dists = np.abs(x_sorted - xi)
            # Truncate to bandwidth
            max_dist = bandwidth * (x_sorted.max() - x_sorted.min())
            mask = dists <= max_dist
            if np.sum(mask) < deg + 1:
                # Include nearest points
                idx_sorted = np.argsort(dists)[:deg + 1]
                mask = np.zeros(n, dtype=bool)
                mask[idx_sorted] = True

            x_local = x_sorted[mask]
            y_local = y_sorted[mask]
            d_local = dists[mask]

            # Tricube weights
            u = d_local / max_dist if max_dist > 0 else np.zeros(len(d_local))
            u = np.clip(u, 0, 1)
            wts = (1 - u**3)**3
            wts = np.maximum(wts, 1e-10)

            # Local polynomial fit
            x_local_scaled = (x_local - xi) / max_dist if max_dist > 0 else x_local
            V = np.vander(x_local_scaled, N=deg + 1, increasing=True)
            W = np.diag(wts)
            try:
                coeff = np.linalg.solve(V.T @ W @ V + 1e-8 * np.eye(deg + 1), V.T @ W @ y_local)
                preds[i] = coeff[0]  # Value at xi is the intercept
            except Exception:
                preds[i] = np.average(y_local, weights=wts)
        return preds

    return predict


def _kfold_cv(fit_func, X, y, k=5, seed=42):
    """Calculate K-Fold CV MSE efficiently."""
    n = len(X)
    if n < k + 2:
        k = max(2, n - 2)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)

    total_err = 0.0
    count = 0
    for f in folds:
        if _remaining() < 3:
            return np.inf
        mask = np.ones(n, dtype=bool)
        mask[f] = False
        X_tr, y_tr = X[mask], y[mask]
        X_te, y_te = X[f], y[f]

        try:
            predict_func = fit_func(X_tr, y_tr)
            preds = predict_func(X_te)
            preds = np.asarray(preds, dtype=np.float64).flatten()
            if len(preds) != len(y_te) or not np.all(np.isfinite(preds)):
                continue
            total_err += np.sum((preds - y_te) ** 2)
            count += len(y_te)
        except Exception:
            return np.inf

    if count == 0:
        return np.inf
    return total_err / count


def _ensemble_predict(models, weights, X):
    """Weighted ensemble of models."""
    preds = np.zeros(len(X))
    for model, w in zip(models, weights):
        preds += w * model(X)
    return preds


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

    x = X[:, 0]
    x_range = x.max() - x.min()
    if x_range < 1e-10:
        x_range = 1.0

    # Build candidate models with CV
    candidates = []

    # RBF with different sigmas
    sigma_fracs = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8]
    for sf in sigma_fracs:
        sigma = sf * x_range
        candidates.append((f"RBF_{sf}", lambda X_tr, y_tr, s=sigma: _fit_rbf_fast(X_tr, y_tr, s)))

    # Polynomials
    for deg in [2, 3, 4, 5, 6, 8, 10]:
        candidates.append((f"Poly_{deg}", lambda X_tr, y_tr, d=deg: _fit_poly_fast(X_tr, y_tr, d)))

    # Chebyshev
    for deg in [3, 5, 7, 9, 11]:
        candidates.append((f"Cheb_{deg}", lambda X_tr, y_tr, d=deg: _fit_chebyshev(X_tr, y_tr, d)))

    # Fourier
    for nt in [2, 3, 4, 5, 7, 10]:
        candidates.append((f"Fourier_{nt}", lambda X_tr, y_tr, nt=nt: _fit_fourier_fast(X_tr, y_tr, nt)))

    # Splines
    for sf in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 20.0]:
        candidates.append((f"Spline_{sf}", lambda X_tr, y_tr, s=sf: _fit_spline_fast(X_tr, y_tr, s)))

    # LOESS with different bandwidths
    for bw in [0.1, 0.2, 0.3, 0.5]:
        candidates.append((f"LOESS_{bw}", lambda X_tr, y_tr, b=bw: _fit_loess(X_tr, y_tr, b, deg=2)))

    # Evaluate with CV, keep track of best
    best_cv_err = np.inf
    best_model = None
    best_name = ""

    # Sort by priority: RBF and spline first
    def _priority(name):
        if name.startswith('RBF'):
            return 0
        if name.startswith('Spline'):
            return 1
        if name.startswith('LOESS'):
            return 2
        if name.startswith('Cheb'):
            return 3
        if name.startswith('Poly'):
            return 4
        if name.startswith('Fourier'):
            return 5
        return 6

    candidates.sort(key=lambda c: _priority(c[0]))

    for name, fit_func in candidates:
        if _remaining() < 5:
            break
        try:
            cv_err = _kfold_cv(fit_func, X, y, k=5)
            if cv_err < best_cv_err:
                best_cv_err = cv_err
                best_model = fit_func(X, y)
                best_name = name
        except Exception:
            pass

    # If no model selected, fallback
    if best_model is None:
        A = np.column_stack([np.ones(n), X[:, 0]])
        try:
            w = np.linalg.lstsq(A, y, rcond=None)[0]

            def predict(X_new):
                An = np.column_stack([np.ones(len(X_new)), X_new[:, 0].astype(np.float64)])
                return An @ w

            best_model = predict
        except Exception:
            c = np.mean(y)

            def predict(X_new):
                return np.full(len(X_new), c)

            best_model = predict

    def safe_predict(X_new):
        try:
            r = best_model(np.asarray(X_new, dtype=np.float64))
            r = np.asarray(r, dtype=np.float64).flatten()
            r = np.nan_to_num(r, nan=0.0, posinf=1e6, neginf=-1e6)
            return r
        except Exception:
            return np.zeros(len(X_new))

    return safe_predict
