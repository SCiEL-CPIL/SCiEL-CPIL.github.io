"""Symbolic regression via rich basis + ridge regression with L-curve/LOOCV selection."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = None
_TIME_LIMIT = 330.0
_SAFETY = 15.0


def _time_left():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _TIME_LIMIT - _SAFETY
    return _DEADLINE - time.time()


def _chebyshev_terms(x, max_deg):
    """Return list of Chebyshev polynomials T_0..T_max_deg evaluated at x (numpy arrays)."""
    n = x.shape[0]
    t0 = np.ones(n)
    t1 = x.copy()
    out = [t0, t1]
    for k in range(2, max_deg + 1):
        tk = 2.0 * x * t1 - t0
        out.append(tk)
        t0, t1 = t1, tk
    return out


def _build_basis(x, max_poly, n_fourier):
    """Build a rich, well-scaled basis for 1-D input x (numpy array)."""
    n = x.shape[0]
    cols = [np.ones(n)]
    # Chebyshev polynomials up to max_poly
    for t in _chebyshev_terms(x, max_poly):
        cols.append(t)
    # Fourier terms (sine/cosine) for periodic / oscillatory components
    for k in range(1, n_fourier + 1):
        cols.append(np.sin(k * x))
        cols.append(np.cos(k * x))
    return np.column_stack(cols)


def _loocv_ridge(B, y, lam):
    """Efficient LOOCV MSE for ridge regression using the hat-matrix formula."""
    n, p = B.shape
    # Solve (B^T B + lam I) w = B^T y
    A = B.T @ B + lam * np.eye(p)
    try:
        w = np.linalg.solve(A, B.T @ y)
    except np.linalg.LinAlgError:
        return np.inf
    yhat = B @ w
    # hat matrix H = B (B^T B + lam I)^-1 B^T ; leverage h_ii
    B_Ainv = B @ np.linalg.inv(A)
    h = np.einsum('ij,ij->i', B, B_Ainv)
    denom = 1.0 - h
    # guard against zero
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    resid = y - yhat
    loocv = np.mean((resid / denom) ** 2)
    return loocv


def _predict_from_basis(B_train, w, X):
    B = _build_basis(X[:, 0], max_poly, n_fourier)
    return B @ w


def fit(X_train, y_train):
    global max_poly, n_fourier
    x = X_train[:, 0].astype(np.float64)
    y = y_train.astype(np.float64)
    n = x.shape[0]

    # Determine basis size from input range
    lo, hi = float(np.min(x)), float(np.max(x))
    span = max(hi - lo, 1e-6)

    # Base configuration
    max_poly = 18
    n_fourier = 8

    B = _build_basis(x, max_poly, n_fourier)
    p = B.shape[1]

    # L-curve / LOOCV sweep over regularization strengths
    lam_grid = np.logspace(-8, 8, 24)
    best_lam = 1e-2
    best_score = np.inf

    # Always keep a plain ridge solution as a fallback
    fallback_w = None
    for lam in lam_grid:
        if _time_left() < 5.0:
            break
        A = B.T @ B + lam * np.eye(p)
        try:
            w = np.linalg.solve(A, B.T @ y)
        except np.linalg.LinAlgError:
            continue
        if fallback_w is None:
            fallback_w = w
        # LOOCV for model selection (cheap: one solve per lam, reuse A)
        B_Ainv = B @ np.linalg.inv(A)
        h = np.einsum('ij,ij->i', B, B_Ainv)
        yhat = B @ w
        denom = 1.0 - h
        denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        resid = y - yhat
        cv = np.mean((resid / denom) ** 2)
        if cv < best_score:
            best_score = cv
            best_lam = lam

    # Final solve at best lambda
    A = B.T @ B + best_lam * np.eye(p)
    try:
        w = np.linalg.solve(A, B.T @ y)
    except np.linalg.LinAlgError:
        w = fallback_w if fallback_w is not None else np.zeros(p)

    # Safety: replace any non-finite coefficients with 0
    w = np.where(np.isfinite(w), w, 0.0)

    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        Bx = _build_basis(X[:, 0], max_poly, n_fourier)
        out = Bx @ w
        out = np.where(np.isfinite(out), out, 0.0)
        return out

    return predict
# EVOLVE-BLOCK-END
