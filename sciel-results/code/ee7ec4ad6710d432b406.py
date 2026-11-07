"""Combined: multi-kernel ensemble + rich basis + rational + GP optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_TIME_START = [0.0]
_TOTAL_DEADLINE = 305.0

def _remaining():
    return _TOTAL_DEADLINE - (time.time() - _TIME_START[0])

def _rbf_kernel(X, Y, gamma):
    d2 = (np.sum(X ** 2, axis=1)[:, None] + np.sum(Y ** 2, axis=1)[None, :] - 2 * X @ Y.T)
    d2 = np.maximum(d2, 0.0)
    return np.exp(-gamma * d2)

def _poly_kernel(X, Y, degree, gamma, coef0):
    return (gamma * X @ Y.T + coef0) ** degree

def _matern_kernel(X, Y, l, nu=2.5):
    d2 = (np.sum(X ** 2, axis=1)[:, None] + np.sum(Y ** 2, axis=1)[None, :] - 2 * X @ Y.T)
    d2 = np.maximum(d2, 0.0)
    r = np.sqrt(d2 + 1e-12)
    if nu == 2.5:
        return (1 + np.sqrt(5) * r + 5 / 3 * r ** 2) * np.exp(-np.sqrt(5) * r / l)
    elif nu == 1.5:
        return (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r / l)
    else:
        return np.exp(-r / l)

def _rich_basis(X):
    """Rich basis: poly up to degree 3 + sin, cos, exp, reciprocal."""
    n, d = X.shape
    cols = [np.ones(n)]
    for j in range(d):
        cols.append(X[:, j])
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
    if d <= 3:
        for i in range(d):
            for j in range(i, d):
                for k in range(j, d):
                    cols.append(X[:, i] * X[:, j] * X[:, k])
    for j in range(d):
        cols.append(np.sin(X[:, j]))
        cols.append(np.cos(X[:, j]))
        x_scaled = (X[:, j] - np.mean(X[:, j])) / (np.std(X[:, j]) + 1e-8)
        cols.append(np.exp(np.clip(x_scaled, -4, 4)))
        cols.append(1.0 / (np.abs(X[:, j]) + 0.1))
        cols.append(np.sin(np.pi * X[:, j]))
        cols.append(np.cos(np.pi * X[:, j]))
    return np.column_stack(cols)

def _fit_ridge_features(F, y, lam):
    """Fit ridge regression on precomputed feature matrix F."""
    n, m = F.shape
    A = F.T @ F + lam * np.eye(m)
    try:
        w = np.linalg.solve(A, F.T @ y)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, F.T @ y, rcond=None)[0]
    return w

def _fit_rbf_ridge(X, y, gamma, lam):
    K = _rbf_kernel(X, X, gamma)
    n = K.shape[0]
    try:
        alpha = np.linalg.solve(K + lam * np.eye(n), y)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(K + lam * np.eye(n), y, rcond=None)[0]
    def predict(Xt):
        return _rbf_kernel(Xt, X, gamma) @ alpha
    return predict

def _fit_gp_m52(X, y):
    """Matern-5/2 GP with Nelder-Mead hyperparameter optimization."""
    n, d = X.shape
    x_range = np.max(np.ptp(X, axis=0)) + 1e-6
    yvar = np.var(y) + 1e-9

    def build_K(theta):
        l = max(theta[0], 1e-4)
        sf = max(theta[1], 1e-6)
        sn = max(theta[2], 1e-8)
        d2 = (np.sum(X ** 2, axis=1)[:, None] + np.sum(X ** 2, axis=1)[None, :] - 2 * X @ X.T)
        d2 = np.maximum(d2, 0.0)
        r = np.sqrt(d2 + 1e-12)
        K = (sf ** 2) * (1 + np.sqrt(5) * r + 5 / 3 * r ** 2) * np.exp(-np.sqrt(5) * r / l)
        K[np.diag_indices(n)] += sn ** 2
        return K

    def neg_lmc(theta):
        try:
            K = build_K(theta)
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            lmc = 0.5 * np.sum(y * alpha) + np.sum(np.log(np.diag(L))) + 0.5 * n * np.log(2 * np.pi)
            return -lmc
        except Exception:
            return 1e10

    theta0 = np.array([x_range, np.sqrt(yvar), np.sqrt(yvar) * 0.01])
    res = minimize(neg_lmc, theta0, method='Nelder-Mead',
                   options={'maxiter': 80, 'xatol': 1e-4, 'fatol': 1e-4})
    l = max(res.x[0], 1e-4)
    sf = max(res.x[1], 1e-6)
    sn = max(res.x[2], 1e-8)
    K = build_K(res.x)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))

    def predict(Xt):
        d2 = (np.sum(Xt ** 2, axis=1)[:, None] + np.sum(X ** 2, axis=1)[None, :] - 2 * Xt @ X.T)
        d2 = np.maximum(d2, 0.0)
        r = np.sqrt(d2 + 1e-12)
        Ks = (sf ** 2) * (1 + np.sqrt(5) * r + 5 / 3 * r ** 2) * np.exp(-np.sqrt(5) * r / l)
        return Ks @ alpha
    return predict

def _fit_rational(X, y):
    """Fit rational function: ratio of two linear basis expansions."""
    n, d = X.shape
    B = np.column_stack([np.ones(n), X])
    m = B.shape[1]

    def rational_pred(coeffs, Xt):
        cnum = coeffs[:m]
        cden = coeffs[m:]
        num = Xt @ cnum
        den = Xt @ cden
        den = np.where(np.abs(den) < 1e-6, np.sign(den + 1e-12) * 1e-6, den)
        return num / den

    x0 = np.zeros(2 * m)
    x0[m] = 1.0

    def obj(coeffs):
        pred = rational_pred(coeffs, X)
        return np.sum((pred - y) ** 2) + 1e-6 * np.sum(coeffs ** 2)

    res = minimize(obj, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-5})
    p = lambda Xt, _c=res.x: rational_pred(_c, Xt)
    return p

def _cv_score(predict_fn, X, y, n_folds=5):
    """K-fold cross-validation MSE."""
    n = len(y)
    idx = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    fold_size = max(1, n // n_folds)
    se = 0.0
    count = 0
    for i in range(n_folds):
        val_idx = idx[i * fold_size:(i + 1) * fold_size]
        tr_idx = np.concatenate([idx[:i * fold_size], idx[(i + 1) * fold_size:]])
        if len(tr_idx) < 5:
            continue
        try:
            pred_val = predict_fn(X[val_idx])
            pred_val = np.asarray(pred_val, dtype=float).ravel()
            if len(pred_val) != len(val_idx):
                continue
            se += np.sum((pred_val - y[val_idx]) ** 2)
            count += len(val_idx)
        except Exception:
            continue
    if count == 0:
        return np.inf
    return se / count

def fit(X_train, y_train):
    _TIME_START[0] = time.time()
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    n, d = X.shape
    yvar = np.var(y) + 1e-12

    if yvar < 1e-12:
        c = float(np.mean(y))
        return lambda Xt: np.full(len(np.atleast_2d(Xt)), c)

    candidates = []

    # 1. Rich basis ridge regression
    if _remaining() > 10:
        try:
            F = _rich_basis(X)
            for lam in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
                if _remaining() < 3:
                    break
                w = _fit_ridge_features(F, y, lam)
                def _pred(Xt, _w=w, _d=d):
                    Ft = _rich_basis(Xt)
                    return Ft @ _w
                candidates.append(('rich_ridge', _pred))
        except Exception:
            pass

    # 2. RBF kernel ridge with various bandwidths
    if _remaining() > 10:
        x_range = np.max(np.ptp(X, axis=0)) + 1e-6
        for gamma in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
            if _remaining() < 3:
                break
            try:
                p = _fit_rbf_ridge(X, y, gamma, 1e-3)
                candidates.append((f'rbf_{gamma}', p))
            except Exception:
                continue

    # 3. Polynomial kernels
    if _remaining() > 5:
        for deg in (2, 3):
            for gamma in (0.5, 1.0, 2.0):
                if _remaining() < 2:
                    break
                try:
                    K_fn = lambda Xa, Xb, dg=deg, g=gamma: _poly_kernel(Xa, Xb, dg, g, 1.0)
                    K = K_fn(X, X)
                    n_k = K.shape[0]
                    try:
                        alpha = np.linalg.solve(K + 1e-3 * np.eye(n_k), y)
                    except np.linalg.LinAlgError:
                        alpha = np.linalg.lstsq(K + 1e-3 * np.eye(n_k), y, rcond=None)[0]
                    def _poly_pred(Xt, _K_fn=K_fn, _alpha=alpha, _X=X):
                        return _K_fn(Xt, _X) @ _alpha
                    candidates.append((f'poly{deg}_g{gamma}', _poly_pred))
                except Exception:
                    continue

    # 4. GP with optimized hyperparameters
    if _remaining() > 20:
        try:
            p = _fit_gp_m52(X, y)
            candidates.append(('gp_m52', p))
        except Exception:
            pass

    # 5. Rational function
    if _remaining() > 15:
        try:
            p = _fit_rational(X, y)
            candidates.append(('rational', p))
        except Exception:
            pass

    # 6. Linear baseline
    try:
        X_lin = np.column_stack([np.ones(n), X])
        w = np.linalg.lstsq(X_lin, y, rcond=None)[0]
        p = lambda Xt: np.column_stack([np.ones(len(Xt)), Xt]) @ w
        candidates.append(('linear', p))
    except Exception:
        pass

    # 7. Quadratic basis
    try:
        X_quad = np.column_stack([np.ones(n)])
        for i in range(d):
            X_quad = np.column_stack([X_quad, X[:, i]])
        for i in range(d):
            for j in range(i, d):
                X_quad = np.column_stack([X_quad, X[:, i] * X[:, j]])
        w = np.linalg.lstsq(X_quad, y, rcond=None)[0]
        def _quad_pred(Xt):
            Xtq = np.column_stack([np.ones(len(Xt))])
            for i in range(d):
                Xtq = np.column_stack([Xtq, Xt[:, i]])
            for i in range(d):
                for j in range(i, d):
                    Xtq = np.column_stack([Xtq, Xt[:, i] * Xt[:, j]])
            return Xtq @ w
        candidates.append(('quad', _quad_pred))
    except Exception:
        pass

    if not candidates:
        c = float(np.mean(y))
        return lambda Xt: np.full(len(np.atleast_2d(Xt)), c)

    # Score all candidates with CV
    scored = []
    for name, p in candidates:
        if _remaining() < 2:
            break
        try:
            score = _cv_score(p, X, y, n_folds=5)
            if np.isfinite(score):
                scored.append((name, p, score))
        except Exception:
            continue

    if not scored:
        c = float(np.mean(y))
        return lambda Xt: np.full(len(np.atleast_2d(Xt)), c)

    scored.sort(key=lambda x: x[2])

    # Ensemble: top-K with diversity-aware weighting
    K_top = min(6, len(scored))
    top = scored[:K_top]

    # Ensure diversity: if multiple kernel models are close, keep at most 2
    selected = [top[0]]
    for item in top[1:]:
        if len(selected) >= K_top:
            break
        # Check similarity to already selected
        is_similar = False
        for sel in selected:
            if (sel[0].startswith('rbf') and item[0].startswith('rbf')) or \
               (sel[0].startswith('poly') and item[0].startswith('poly')) or \
               (sel[0].startswith('rich') and item[0].startswith('rich')):
                is_similar = True
                break
        if not is_similar:
            selected.append(item)

    # If we didn't get enough diverse, fill with remaining top
    if len(selected) < min(3, len(scored)):
        for item in top:
            if item not in selected and len(selected) < min(3, len(scored)):
                selected.append(item)

    scores = np.array([s[2] for s in selected])
    # Inverse-variance weighting with floor
    weights = 1.0 / (scores + 1e-6)
    weights = weights / weights.sum()

    predictors = [s[1] for s in selected]
    mean_y = float(np.mean(y))

    def predict(Xt):
        Xt = np.asarray(Xt, dtype=float)
        if Xt.ndim == 1:
            Xt = Xt.reshape(1, -1)
        out = np.zeros(len(Xt))
        valid_count = 0
        for p, w in zip(predictors, weights):
            try:
                val = np.asarray(p(Xt), dtype=float).ravel()
                if len(val) == len(Xt) and np.all(np.isfinite(val)):
                    out += w * val
                    valid_count += 1
            except Exception:
                continue
        if valid_count == 0:
            out = np.full(len(Xt), mean_y)
        else:
            out = np.where(np.isfinite(out), out, mean_y)
        return out

    return predict
# EVOLVE-BLOCK-END
