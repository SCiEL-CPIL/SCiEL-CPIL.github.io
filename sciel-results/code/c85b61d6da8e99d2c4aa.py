"""Symbolic regression via sparse linear model over a rich basis, with kernel fallback."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.linalg import lstsq as _lstsq


def _safe_predict(model, X, fallback):
    try:
        p = model.predict(X)
        p = np.asarray(p, dtype=float).ravel()
        if p.shape[0] != X.shape[0] or not np.all(np.isfinite(p)):
            return fallback.predict(X)
        return p
    except Exception:
        return fallback.predict(X)


def _make_poly_features(X, deg):
    d = X.shape[1]
    cols = [np.ones(len(X))]
    for total in range(1, deg + 1):
        for idx in _compositions(d, total):
            term = np.ones(len(X))
            for k in range(d):
                if idx[k] > 0:
                    term = term * np.power(X[:, k], idx[k])
            cols.append(term)
    return np.column_stack(cols)


def _compositions(d, total):
    if d == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(d - 1, total - first):
            yield (first,) + rest


def _make_rich_features(X, poly_deg=4):
    d = X.shape[1]
    n = len(X)
    cols = [np.ones(n)]
    # linear
    for j in range(d):
        cols.append(X[:, j])
    # polynomial terms up to poly_deg
    for total in range(2, poly_deg + 1):
        for idx in _compositions(d, total):
            term = np.ones(n)
            for k in range(d):
                if idx[k] > 0:
                    term = term * np.power(X[:, k], idx[k])
            cols.append(term)
    # trig
    for j in range(d):
        x = X[:, j]
        cols.append(np.sin(x))
        cols.append(np.cos(x))
        cols.append(np.sin(2 * x))
        cols.append(np.cos(2 * x))
    # exponential (guarded)
    for j in range(d):
        x = X[:, j]
        cols.append(np.exp(np.clip(x, -10, 10)))
        cols.append(np.exp(np.clip(-x, -10, 10)))
    # pairwise products of distinct inputs
    for i in range(d):
        for j in range(i + 1, d):
            cols.append(X[:, i] * X[:, j])
    # sqrt
    for j in range(d):
        cols.append(np.sqrt(np.abs(X[:, j])))
    return np.column_stack(cols)


def _fit_sparse(X, y, time_budget):
    """Fit a sparse linear model via LARS; returns (predict_fn, score) or None."""
    from sklearn.linear_model import lars_path
    Xf = _make_rich_features(X, poly_deg=4)
    # standardize columns to help LARS
    mu = Xf.mean(axis=0)
    sd = Xf.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Xf_s = (Xf - mu) / sd
    y_mu = y.mean()
    y_sd = y.std()
    y_s = (y - y_mu) / (y_sd if y_sd > 1e-12 else 1.0)

    try:
        coefs, _, _, _, _, _, _ = lars_path(Xf_s, y_s, n_nonzero_coefs=15, method='lars')
    except Exception:
        return None

    # pick the sparsest model with low training error
    best = None
    for k in range(1, 15):
        coef = coefs[k]
        pred_s = Xf_s @ coef + y_mu
        pred = pred_s * y_sd + y_mu
        err = np.mean((pred - y) ** 2)
        var = np.var(y)
        nrmse = err / (var if var > 1e-12 else 1.0)
        if best is None or nrmse < best[1]:
            best = (coef, nrmse, k)
    if best is None:
        return None
    coef, nrmse, k = best
    # reconstruct in original scale
    # Xf_s = (Xf - mu)/sd ; pred_s = Xf_s @ coef ; pred = pred_s*y_sd + y_mu
    # => pred = (Xf @ coef / sd - mu @ coef / sd) * y_sd + y_mu
    coef_orig = coef / sd
    intercept = (-mu @ coef / sd) * y_sd + y_mu

    def predict(Xq):
        Xf_q = _make_rich_features(Xq, poly_deg=4)
        return Xf_q @ coef_orig + intercept

    return (predict, nrmse)


def _fit_kernel(X, y, time_budget):
    """Kernel ridge regression with tuned bandwidth; returns predict fn."""
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.model_selection import KFold
    n = len(X)
    # simple bandwidth search
    best_kr = None
    best_score = np.inf
    candidates = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    alphas = [1e-3, 1e-2, 1e-1, 1.0]
    kf = KFold(n_splits=3, shuffle=True, random_state=0)
    for gamma in candidates:
        for alpha in alphas:
            try:
                scores = []
                for tr, va in kf.split(X):
                    kr = KernelRidge(alpha=alpha, kernel='rbf', gamma=gamma)
                    kr.fit(X[tr], y[tr])
                    p = kr.predict(X[va])
                    scores.append(np.mean((p - y[va]) ** 2))
                s = np.mean(scores)
                if s < best_score:
                    best_score = s
                    best_kr = (gamma, alpha)
            except Exception:
                continue
    if best_kr is None:
        gamma, alpha = 1.0, 1e-2
    else:
        gamma, alpha = best_kr
    kr = KernelRidge(alpha=alpha, kernel='rbf', gamma=gamma)
    kr.fit(X, y)
    return kr


def fit(X_train, y_train):
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float).ravel()
    start = time.time()
    deadline = start + 250.0  # safety margin before 300s

    # Always build a polynomial least-squares baseline (fast, robust)
    def _poly_predict(Xq):
        Xq = np.asarray(Xq, dtype=float)
        d = Xq.shape[1]
        cols = [np.ones(len(Xq))] + [Xq[:, j] for j in range(d)] + \
               [Xq[:, i] * Xq[:, j] for i in range(d) for j in range(i, d)]
        F = np.column_stack(cols)
        w, *_ = np.linalg.lstsq(F, y, rcond=None)
        return F @ w

    # Baseline: degree-2 poly
    baseline = _poly_predict(X)
    base_nrmse = np.mean((baseline - y) ** 2) / (np.var(y) if np.var(y) > 1e-12 else 1.0)

    # Try sparse symbolic regression
    best_predict = lambda Xq: _poly_predict(Xq)
    best_nrmse = base_nrmse
    if time.time() < deadline - 30:
        try:
            res = _fit_sparse(X, y, deadline - time.time())
            if res is not None:
                pred_fn, nrmse = res
                p = pred_fn(X)
                if np.all(np.isfinite(p)):
                    if nrmse < best_nrmse:
                        best_predict = pred_fn
                        best_nrmse = nrmse
        except Exception:
            pass

    # Try kernel ridge if time remains and it might help
    if time.time() < deadline - 20:
        try:
            kr = _fit_kernel(X, y, deadline - time.time())
            p = kr.predict(X)
            nrmse = np.mean((p - y) ** 2) / (np.var(y) if np.var(y) > 1e-12 else 1.0)
            if np.all(np.isfinite(p)) and nrmse < best_nrmse:
                best_predict = lambda Xq, _kr=kr: _kr.predict(Xq)
                best_nrmse = nrmse
        except Exception:
            pass

    # Final predict with safety
    def predict(Xq):
        Xq = np.asarray(Xq, dtype=float)
        out = best_predict(Xq)
        out = np.asarray(out, dtype=float).ravel()
        if out.shape[0] != Xq.shape[0] or not np.all(np.isfinite(out)):
            out = _poly_predict(Xq)
        return out
    return predict
# EVOLVE-BLOCK-END
