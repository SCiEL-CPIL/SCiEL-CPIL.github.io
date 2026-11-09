"""Rewritten program: multi-basis symbolic regression with adaptive selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _safe_predict(X, model, fallback_val):
    """Call model.predict, returning fallback on any error or non-finite output."""
    try:
        out = model.predict(X)
        out = np.asarray(out, dtype=float)
        if out.shape[0] != X.shape[0]:
            raise ValueError("shape mismatch")
        if not np.all(np.isfinite(out)):
            raise ValueError("non-finite")
        return out
    except Exception:
        return np.full(X.shape[0], fallback_val)

def _fit_ridge(X, y, lam):
    """Fit ridge regression; returns (w, intercept)."""
    n = X.shape[0]
    Xb = np.hstack([X, np.ones((n, 1))])
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    b = Xb.T @ y
    try:
        sol = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(A, b, rcond=None)[0]
    return sol

def _make_poly_features(X, deg):
    """Full polynomial feature expansion up to degree deg (with interactions)."""
    n, d = X.shape
    cols = [np.ones(n)]
    for total_deg in range(1, deg + 1):
        for exponents in product(range(total_deg + 1), repeat=d):
            if sum(exponents) != total_deg:
                continue
            term = np.ones(n)
            for j, e in enumerate(exponents):
                if e > 0:
                    term = term * (X[:, j] ** e)
            cols.append(term)
    return np.column_stack(cols)

def _make_chebyshev_features(X, deg):
    """Chebyshev polynomial features per dimension (tensor product)."""
    n, d = X.shape
    # Normalize each dimension to [-1, 1]
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    Xn = (X - mins) / ranges * 2.0 - 1.0
    # Per-dimension Chebyshev up to deg
    cheb = [np.ones(n)]
    if deg >= 1:
        cheb.append(Xn.copy())
    for k in range(2, deg + 1):
        cheb.append(2.0 * Xn * cheb[-1] - cheb[-2])
    # Tensor product: pick one cheb level per dimension
    cols = []
    for idx in product(range(len(cheb)), repeat=d):
        term = np.ones(n)
        for j, k in enumerate(idx):
            term = term * cheb[k][:, j] if cheb[k].ndim > 1 else term * cheb[k]
        cols.append(term)
    return np.column_stack(cols), (mins, ranges)

def _make_rbf_features(X, centers, gammas):
    """RBF features with given centers and gammas."""
    n = X.shape[0]
    cols = []
    for c, g in zip(centers, gammas):
        d2 = np.sum((X - c) ** 2, axis=1)
        cols.append(np.exp(-g * d2))
    return np.column_stack(cols)

def _make_fourier_features(X, freqs):
    """sin/cos Fourier features."""
    n, d = X.shape
    cols = [np.ones(n)]
    for f in freqs:
        for j in range(d):
            cols.append(np.sin(f * X[:, j]))
            cols.append(np.cos(f * X[:, j]))
    return np.column_stack(cols)

def fit(X_train, y_train):
    start_time = time.time()
    DEADLINE = start_time + 290.0  # margin before 330s kill

    n, d = X_train.shape
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train))
    if y_std < 1e-12:
        return lambda X: np.full(X.shape[0], y_mean)

    def time_ok():
        return time.time() < DEADLINE

    # Validation split
    idx = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(idx)
    n_val = max(20, int(0.2 * n))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    candidates = []  # list of (val_mse, predict_fn)

    def add_candidate(predict_fn):
        try:
            pv = predict_fn(X_val)
            pv = np.asarray(pv, dtype=float)
            if pv.shape[0] != X_val.shape[0] or not np.all(np.isfinite(pv)):
                return
            mse = float(np.mean((pv - y_val) ** 2))
            candidates.append((mse, predict_fn))
        except Exception:
            pass

    # ---- 1. Polynomial features of varying degree ----
    for deg in [1, 2, 3, 4]:
        if not time_ok():
            break
        try:
            F_tr = _make_poly_features(X_tr, deg)
            F_val = _make_poly_features(X_val, deg)
            # ridge with small lambda
            w = _fit_ridge(F_tr, y_tr, 1e-3)
            def mkpred(Fv, w=w):
                return lambda X: _safe_predict(X, type('M', (), {'predict': lambda self, Xx: Fv if Xx is X_val else None}), y_mean)
            # Simpler: precompute mapping not possible; just store feature builder
            def predict_poly(Xx, Fb=lambda Xx: _make_poly_features(Xx, deg), w=w):
                try:
                    Fx = Fb(Xx)
                    out = Fx @ w
                    if not np.all(np.isfinite(out)):
                        raise ValueError
                    return out
                except Exception:
                    return np.full(Xx.shape[0], y_mean)
            add_candidate(predict_poly)
        except Exception:
            pass

    # ---- 2. Chebyshev features ----
    for deg in [2, 3, 4, 5]:
        if not time_ok():
            break
        try:
            F_tr, norm = _make_chebyshev_features(X_tr, deg)
            F_val, _ = _make_chebyshev_features(X_val, deg)
            w = _fit_ridge(F_tr, y_tr, 1e-3)
            def predict_cheb(Xx, w=w, deg=deg):
                try:
                    Fx, _ = _make_chebyshev_features(Xx, deg)
                    out = Fx @ w
                    if not np.all(np.isfinite(out)):
                        raise ValueError
                    return out
                except Exception:
                    return np.full(Xx.shape[0], y_mean)
            add_candidate(predict_cheb)
        except Exception:
            pass

    # ---- 3. Fourier features with several frequency sets ----
    # Estimate characteristic scale from data range
    ranges = X_tr.max(axis=0) - X_tr.min(axis=0)
    ranges[ranges == 0] = 1.0
    base_freq = 1.0 / np.mean(ranges)
    for mult in [1, 2, 4, 8]:
        if not time_ok():
            break
        try:
            freqs = [base_freq * mult * k for k in range(1, 5)]
            F_tr = _make_fourier_features(X_tr, freqs)
            w = _fit_ridge(F_tr, y_tr, 1e-3)
            def predict_fourier(Xx, w=w, freqs=freqs):
                try:
                    Fx = _make_fourier_features(Xx, freqs)
                    out = Fx @ w
                    if not np.all(np.isfinite(out)):
                        raise ValueError
                    return out
                except Exception:
                    return np.full(Xx.shape[0], y_mean)
            add_candidate(predict_fourier)
        except Exception:
            pass

    # ---- 4. RBF with grid of centers ----
    if time_ok():
        try:
            # Choose centers on a grid
            n_grid = max(3, int(round(n ** 0.5)))
            mins = X_tr.min(axis=0)
            maxs = X_tr.max(axis=0)
            ranges = maxs - mins
            ranges[ranges == 0] = 1.0
            grid_pts = [np.linspace(mins[j], maxs[j], n_grid) for j in range(d)]
            centers = []
            for combo in product(*grid_pts):
                centers.append(np.array(combo))
            # Limit number of centers
            if len(centers) > 200:
                # subsample
                sel = np.random.choice(len(centers), 200, replace=False)
                centers = [centers[i] for i in sel]
            for g_mult in [1.0, 5.0, 20.0]:
                if not time_ok():
                    break
                gammas = [g_mult / (d * np.mean(ranges ** 2)) for _ in centers]
                F_tr = _make_rbf_features(X_tr, centers, gammas)
                w = _fit_ridge(F_tr, y_tr, 1e-2)
                def predict_rbf(Xx, w=w, centers=centers, gammas=gammas):
                    try:
                        Fx = _make_rbf_features(Xx, centers, gammas)
                        out = Fx @ w
                        if not np.all(np.isfinite(out)):
                            raise ValueError
                        return out
                    except Exception:
                        return np.full(Xx.shape[0], y_mean)
                add_candidate(predict_rbf)
        except Exception:
            pass

    # ---- 5. Scaled polynomial (normalize inputs first) ----
    if time_ok():
        try:
            mins = X_tr.min(axis=0)
            maxs = X_tr.max(axis=0)
            ranges = maxs - mins
            ranges[ranges == 0] = 1.0
            def norm(Xx):
                return (Xx - mins) / ranges
            for deg in [2, 3]:
                if not time_ok():
                    break
                F_tr = _make_poly_features(norm(X_tr), deg)
                w = _fit_ridge(F_tr, y_tr, 1e-3)
                def predict_normpoly(Xx, w=w, deg=deg):
                    try:
                        Fx = _make_poly_features(norm(Xx), deg)
                        out = Fx @ w
                        if not np.all(np.isfinite(out)):
                            raise ValueError
                        return out
                    except Exception:
                        return np.full(Xx.shape[0], y_mean)
                add_candidate(predict_normpoly)
        except Exception:
            pass

    # Always include a constant baseline
    add_candidate(lambda X: np.full(X.shape[0], y_mean))

    # Select best by validation MSE
    if not candidates:
        return lambda X: np.full(X.shape[0], y_mean)

    candidates.sort(key=lambda t: t[0])
    best_mse, best_pred = candidates[0]

    # Final predict: use best model, with safety fallback
    def final_predict(X):
        out = _safe_predict(X, type('M', (), {'predict': lambda self, Xx: best_pred(Xx)}), y_mean)
        return out

    return final_predict
# EVOLVE-BLOCK-END
