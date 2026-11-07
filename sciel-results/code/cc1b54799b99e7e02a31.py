import numpy as np
import time
import warnings
from scipy.interpolate import CubicSpline, UnivariateSpline

warnings.filterwarnings('ignore')

_START_TIME = None
TIME_LIMIT = 330.0
SAFETY_MARGIN = 30.0

def _time_left():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        return TIME_LIMIT - SAFETY_MARGIN
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)

# ---------------- RBF kernel / ridge / LOOCV ----------------
def _rbf_kernel(X1, X2, length_scale):
    X1 = np.asarray(X1, dtype=np.float64).reshape(-1, 1)
    X2 = np.asarray(X2, dtype=np.float64).reshape(-1, 1)
    d2 = np.sum((X1 - X2.T)**2, axis=1)
    return np.exp(-0.5 * d2 / (length_scale**2))

def _fit_ridge_kernel(x_train, y_train, x_test, length_scale, alpha):
    K = _rbf_kernel(x_train, x_train, length_scale)
    n = len(x_train)
    K_reg = K + alpha * np.eye(n)
    K_test = _rbf_kernel(x_test, x_train, length_scale)
    try:
        w = np.linalg.solve(K_reg, y_train)
        return K_test @ w
    except np.linalg.LinAlgError:
        w, _, _, _ = np.linalg.lstsq(K_reg, y_train, rcond=None)
        return K_test @ w

def _loocv_ridge_fast(K, y, alpha):
    """LOOCV MSE for ridge in RKHS via hat-matrix (no per-fold refits)."""
    n = len(y)
    K_reg = K + alpha * np.eye(n)
    try:
        K_inv = np.linalg.inv(K_reg)
    except np.linalg.LinAlgError:
        return np.inf
    H = K @ K_inv
    Hy = H @ y
    Hdiag = np.diag(H)
    denom = 1.0 - Hdiag
    yhat_loo = np.copy(Hy)
    mask = np.abs(denom) > 1e-12
    yhat_loo[mask] = (Hy[mask] - Hdiag[mask] * y[mask]) / denom[mask]
    resid = yhat_loo - y
    return np.mean(resid * resid)

def _fit_rbf_kernel_ridge_cv(x, y, length_scales, alphas):
    n = len(y)
    best_mse = np.inf
    best_ls = length_scales[0]
    best_alpha = alphas[0]
    for ls in length_scales:
        if _time_left() < 5.0:
            break
        K = _rbf_kernel(x, x, ls)
        for alpha in alphas:
            if _time_left() < 5.0:
                break
            try:
                m = _loocv_ridge_fast(K, y, alpha)
            except Exception:
                m = np.inf
            if m < best_mse:
                best_mse = m
                best_ls = ls
                best_alpha = alpha
    return best_ls, best_alpha, best_mse

# ---------------- Chebyshev basis ----------------
def _build_chebyshev_basis(x, x_min, x_max, n_terms):
    x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
    T = np.zeros((len(x), n_terms))
    if n_terms > 0:
        T[:, 0] = 1.0
    if n_terms > 1:
        T[:, 1] = x_norm
    for k in range(2, n_terms):
        T[:, k] = 2 * x_norm * T[:, k-1] - T[:, k-2]
    return T

def _fit_chebyshev_ridge(x, y, x_test, n_terms, lam, x_min, x_max):
    V = _build_chebyshev_basis(x, x_min, x_max, n_terms)
    A = V.T @ V + lam * np.eye(n_terms)
    try:
        w = np.linalg.solve(A, V.T @ y)
    except np.linalg.LinAlgError:
        w, _, _, _ = np.linalg.lstsq(A, V.T @ y, rcond=None)
    V_test = _build_chebyshev_basis(x_test, x_min, x_max, n_terms)
    return V_test @ w

def _fit_chebyshev_ridge_cv(x, y, n_terms_options, lambdas, n_folds=5):
    n = len(y)
    np.random.seed(42)
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    best_mse = np.inf
    best_n = n_terms_options[0]
    best_lambda = lambdas[0]
    x_min, x_max = np.min(x), np.max(x)
    for n_terms in n_terms_options:
        if _time_left() < 5.0:
            break
        for lam in lambdas:
            if _time_left() < 5.0:
                break
            fold_mses = []
            for i in range(n_folds):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
                try:
                    pred_val = _fit_chebyshev_ridge(x[train_idx], y[train_idx], x[val_idx], n_terms, lam, x_min, x_max)
                    fold_mses.append(np.mean((pred_val - y[val_idx])**2))
                except Exception:
                    fold_mses.append(np.inf)
            mean_mse = np.mean(fold_mses)
            if mean_mse < best_mse:
                best_mse = mean_mse
                best_n = n_terms
                best_lambda = lam
    return best_n, best_lambda, best_mse

# ---------------- Rich combined basis (Chebyshev + Fourier + exponential) ----------------
def _build_rich_basis(x, x_min, x_max, n_poly, n_fourier, n_exp):
    n = len(x)
    total = n_poly + 2 * n_fourier + n_exp
    B = np.zeros((n, total))
    if n_poly > 0:
        B[:, :n_poly] = _build_chebyshev_basis(x, x_min, x_max, n_poly)
    L = x_max - x_min
    if L <= 0:
        L = 1.0
    col = n_poly
    for k in range(1, n_fourier + 1):
        arg = k * np.pi * (x - x_min) / L
        if col < total:
            B[:, col] = np.cos(arg); col += 1
        if col < total:
            B[:, col] = np.sin(arg); col += 1
    col = n_poly + 2 * n_fourier
    for k in range(n_exp):
        if col < total:
            rate = (k + 1) * 0.5 / L
            B[:, col] = np.exp(rate * (x - x_min)); col += 1
    return B

def _fit_rich_basis_ridge(x, y, x_test, n_poly, n_fourier, n_exp, lam, x_min, x_max):
    B = _build_rich_basis(x, x_min, x_max, n_poly, n_fourier, n_exp)
    n_terms = B.shape[1]
    A = B.T @ B + lam * np.eye(n_terms)
    try:
        w = np.linalg.solve(A, B.T @ y)
    except np.linalg.LinAlgError:
        w, _, _, _ = np.linalg.lstsq(A, B.T @ y, rcond=None)
    B_test = _build_rich_basis(x_test, x_min, x_max, n_poly, n_fourier, n_exp)
    return B_test @ w

def _fit_rich_basis_cv(x, y, configs, lambdas, n_folds=5):
    n = len(y)
    np.random.seed(42)
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    best_mse = np.inf
    best_config = configs[0]
    best_lambda = lambdas[0]
    x_min, x_max = np.min(x), np.max(x)
    for config in configs:
        if _time_left() < 5.0:
            break
        for lam in lambdas:
            if _time_left() < 5.0:
                break
            fold_mses = []
            for i in range(n_folds):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
                try:
                    pred_val = _fit_rich_basis_ridge(x[train_idx], y[train_idx], x[val_idx],
                                                     config[0], config[1], config[2], lam, x_min, x_max)
                    fold_mses.append(np.mean((pred_val - y[val_idx])**2))
                except Exception:
                    fold_mses.append(np.inf)
            mean_mse = np.mean(fold_mses)
            if mean_mse < best_mse:
                best_mse = mean_mse
                best_config = config
                best_lambda = lam
    return best_config, best_lambda, best_mse

# ---------------- Smoothing spline ----------------
def _spline_fit(x, y, x_test, s):
    try:
        order = np.argsort(x)
        x_sorted = x[order]; y_sorted = y[order]
        if s <= 0:
            s = 1e-8 * max(np.var(y), 1e-10)
        spl = UnivariateSpline(x_sorted, y_sorted, k=3, s=s)
        return spl(x_test)
    except Exception:
        return np.full(len(x_test), np.mean(y))

def _spline_cv(x, y, s_options, n_folds=5):
    n = len(y)
    np.random.seed(42)
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    best_mse = np.inf
    best_s = s_options[0]
    for s in s_options:
        if _time_left() < 5.0:
            break
        fold_mses = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            try:
                pred_val = _spline_fit(x[train_idx], y[train_idx], x[val_idx], s)
                fold_mses.append(np.mean((pred_val - y[val_idx])**2))
            except Exception:
                fold_mses.append(np.inf)
        mean_mse = np.mean(fold_mses)
        if mean_mse < best_mse:
            best_mse = mean_mse
            best_s = s
    return best_s, best_mse

def _fit_cubic_spline(x, y, x_test):
    try:
        order = np.argsort(x)
        spl = CubicSpline(x[order], y[order])
        return spl(x_test)
    except Exception:
        return np.full(len(x_test), np.mean(y))

# ---------------- LOESS (adaptive-bandwidth variant) ----------------
def _loess_predict(x_train, y_train, x_test, frac, degree=1, adaptive=False):
    x_train = np.asarray(x_train, dtype=np.float64).ravel()
    y_train = np.asarray(y_train, dtype=np.float64).ravel()
    x_test = np.asarray(x_test, dtype=np.float64).ravel()
    n = len(x_train)
    m_base = max(2, int(frac * n))
    preds = np.empty(len(x_test))
    # median spacing for adaptive bandwidth
    if adaptive:
        order = np.argsort(x_train)
        xs = x_train[order]
        gaps = np.diff(xs)
        med_gap = np.median(gaps) if len(gaps) > 0 else 1.0
        if med_gap <= 0:
            med_gap = 1e-10
        span = med_gap
    for i, xv in enumerate(x_test):
        if _time_left() < 0.5:
            preds[i:] = np.mean(y_train)
            break
        dists = np.abs(x_train - xv)
        if adaptive:
            # choose m so that the window covers ~span around xv
            m = int(np.sum(dists <= span))
            m = max(2, min(n, m))
        else:
            m = m_base
        if m >= n:
            idx = np.arange(n)
        else:
            idx = np.argpartition(dists, m - 1)[:m]
        max_dist = dists[idx].max()
        if max_dist < 1e-15:
            max_dist = 1e-10
        w = (1 - (dists[idx] / max_dist)**3)**3
        try:
            if degree == 0:
                preds[i] = np.sum(w * y_train[idx]) / np.sum(w)
            else:
                Xw = x_train[idx] - xv
                A = np.vstack([np.ones_like(Xw), Xw]).T
                WA = A * w[:, None]
                try:
                    coef = np.linalg.solve(WA.T @ A + 1e-12 * np.eye(2), WA.T @ y_train[idx])
                    preds[i] = coef[0]
                except np.linalg.LinAlgError:
                    preds[i] = np.sum(w * y_train[idx]) / np.sum(w)
        except Exception:
            preds[i] = np.mean(y_train)
    return preds

def _loess_cv(x, y, frac_options, n_folds=5, adaptive=False):
    n = len(y)
    np.random.seed(42)
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    best_mse = np.inf
    best_frac = frac_options[0]
    for frac in frac_options:
        if _time_left() < 5.0:
            break
        fold_mses = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            try:
                pred_val = _loess_predict(x[train_idx], y[train_idx], x[val_idx], frac, adaptive=adaptive)
                fold_mses.append(np.mean((pred_val - y[val_idx])**2))
            except Exception:
                fold_mses.append(np.inf)
        mean_mse = np.mean(fold_mses)
        if mean_mse < best_mse:
            best_mse = mean_mse
            best_frac = frac
    return best_frac, best_mse

def fit(X_train, y_train):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

    X = np.asarray(X_train, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    n = len(y)
    x = X[:, 0].copy()
    x_min, x_max = np.min(x), np.max(x)

    methods = []

    # 1) RBF kernel + ridge (LOOCV, finer grid)
    try:
        ls_options = np.logspace(-2.5, 1.0, 14)
        alpha_options = np.logspace(-4.0, 1.0, 12)
        best_ls, best_alpha, kernel_mse = _fit_rbf_kernel_ridge_cv(x, y, ls_options, alpha_options)
        methods.append(('kernel', kernel_mse, {'ls': best_ls, 'alpha': best_alpha}))
    except Exception:
        pass

    # 2) Chebyshev + ridge
    try:
        n_terms_options = [5, 8, 12, 16, 20, 25, 30]
        lam_options = np.logspace(-4, 2, 10)
        best_n, best_lam, cheb_mse = _fit_chebyshev_ridge_cv(x, y, n_terms_options, lam_options)
        methods.append(('chebyshev', cheb_mse, {'n': best_n, 'lam': best_lam}))
    except Exception:
        pass

    # 3) Rich combined basis + ridge
    try:
        configs = [(6, 3, 2), (8, 4, 3), (10, 5, 4), (12, 6, 5), (16, 8, 6)]
        lam_options = np.logspace(-3, 2, 8)
        best_cfg, best_lam, rich_mse = _fit_rich_basis_cv(x, y, configs, lam_options)
        methods.append(('rich', rich_mse, {'cfg': best_cfg, 'lam': best_lam}))
    except Exception:
        pass

    # 4) Smoothing spline
    try:
        s_options = np.logspace(-10, -2, 15)
        best_s, spline_mse = _spline_cv(x, y, s_options)
        methods.append(('spline', spline_mse, {'s': best_s}))
    except Exception:
        pass

    # 5) LOESS fixed-fraction
    try:
        frac_options = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        best_frac, loess_mse = _loess_cv(x, y, frac_options, adaptive=False)
        methods.append(('loess', loess_mse, {'frac': best_frac, 'adaptive': False}))
    except Exception:
        pass

    # 6) LOESS adaptive bandwidth
    try:
        frac_options = [0.1, 0.2, 0.3, 0.5]
        best_frac, loess_ad_mse = _loess_cv(x, y, frac_options, adaptive=True)
        methods.append(('loess_adaptive', loess_ad_mse, {'frac': best_frac, 'adaptive': True}))
    except Exception:
        pass

    # 7) Interpolating cubic spline (LOO-ish)
    try:
        np.random.seed(42)
        indices = np.arange(n)
        np.random.shuffle(indices)
        folds = np.array_split(indices, 5)
        fold_mses = []
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            pred_val = _fit_cubic_spline(x[train_idx], y[train_idx], x[val_idx])
            fold_mses.append(np.mean((pred_val - y[val_idx])**2))
        spline_interp_mse = np.mean(fold_mses)
        methods.append(('spline_interp', spline_interp_mse, {}))
    except Exception:
        pass

    if not methods:
        mean_y = float(np.mean(y))
        def predict(X):
            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            return np.full(len(X_arr), mean_y)
        return predict

    methods.sort(key=lambda m: m[1])
    best_name, _, params = methods[0]

    x_store = x.copy()
    y_store = y.copy()
    xmin_store = x_min
    xmax_store = x_max
    n_store = n

    if best_name == 'kernel':
        K = _rbf_kernel(x, x, params['ls'])
        K_reg = K + params['alpha'] * np.eye(n)
        try:
            w = np.linalg.solve(K_reg, y)
        except np.linalg.LinAlgError:
            w, _, _, _ = np.linalg.lstsq(K_reg, y, rcond=None)
        w_store = w.copy(); ls_store = params['ls']
        def predict(X):
            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            xv = X_arr[:, 0]
            K_test = _rbf_kernel(xv, x_store, ls_store)
            return K_test @ w_store

    elif best_name == 'chebyshev':
        V = _build_chebyshev_basis(x, xmin_store, xmax_store, params['n'])
        A = V.T @ V + params['lam'] * np.eye(params['n'])
        try:
            w = np.linalg.solve(A, V.T @ y)
        except np.linalg.LinAlgError:
            w, _, _, _ = np.linalg.lstsq(A, V.T @ y, rcond=None)
        w_store = w.copy(); n_store = params['n']
        def predict(X):
            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            V_test = _build_chebyshev_basis(X_arr[:, 0], xmin_store, xmax_store, n_store)
            return V_test @ w_store

    elif best_name == 'rich':
        cfg = params['cfg']
        B = _build_rich_basis(x, xmin_store, xmax_store, cfg[0], cfg[1], cfg[2])
        nt = B.shape[1]
        A = B.T @ B + params['lam'] * np.eye(nt)
        try:
            w = np.linalg.solve(A, B.T @ y)
        except np.linalg.LinAlgError:
            w, _, _, _ = np.linalg.lstsq(A, B.T @ y, rcond=None)
        w_store = w.copy(); cfg_store = cfg
        def predict(X):
            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            B_test = _build_rich_basis(X_arr[:, 0], xmin_store, xmax_store, cfg_store[0], cfg_store[1], cfg_store[2])
            return B_test @ w_store

    elif best_name == 'spline':
        order = np.argsort(x)
        try:
            spl = UnivariateSpline(x[order], y[order], k=3, s=params['s'])
            def predict(X):
                X_arr = np.asarray(X, dtype=np.float64)
                if X_arr.ndim == 1:
                    X_arr = X_arr.reshape(-1, 1)
                return spl(X_arr[:, 0])
        except Exception:
            mean_y = float(np.mean(y))
            def predict(X):
                X_arr = np.asarray(X, dtype=np.float64)
                if X_arr.ndim == 1:
                    X_arr = X_arr.reshape(-1, 1)
                return np.full(len(X_arr), mean_y)

    elif best_name in ('loess', 'loess_adaptive'):
        frac_store = params['frac']; adaptive_store = params['adaptive']
        def predict(X):
            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            return _loess_predict(x_store, y_store, X_arr[:, 0], frac_store, adaptive=adaptive_store)

    else:  # spline_interp
        order = np.argsort(x)
        try:
            spl = CubicSpline(x[order], y[order])
            def predict(X):
                X_arr = np.asarray(X, dtype=np.float64)
                if X_arr.ndim == 1:
                    X_arr = X_arr.reshape(-1, 1)
                return spl(X_arr[:, 0])
        except Exception:
            mean_y = float(np.mean(y))
            def predict(X):
                X_arr = np.asarray(X, dtype=np.float64)
                if X_arr.ndim == 1:
                    X_arr = X_arr.reshape(-1, 1)
                return np.full(len(X_arr), mean_y)

    # Final safety wrapper: guarantee finite output
    def _safe_predict(X):
        try:
            out = np.asarray(predict(X), dtype=np.float64)
            out = np.where(np.isfinite(out), out, float(np.mean(y)))
            return out
        except Exception:
            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            return np.full(len(X_arr), float(np.mean(y)))

    return _safe_predict
