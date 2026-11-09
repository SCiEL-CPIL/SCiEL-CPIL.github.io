"""Gaussian-process structural change for 1D symbolic regression over 4 hidden equations."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


def _poly_predict(X, y):
    """Fallback: degree-8 polynomial least-squares (always finite)."""
    x = X[:, 0]
    w, *_ = np.linalg.lstsq(x ** np.arange(0, 9)[:, None], y, rcond=None)
    return lambda X2: (X2[:, 0] ** np.arange(0, 9)[:, None]) @ w


def fit(X_train, y_train):
    start = time.time()
    deadline = start + 295.0  # safety margin vs 330s wall-clock

    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    x = X[:, 0].copy()
    n = x.size

    # Normalize inputs to a unit-ish scale so length-scale optimization is well-conditioned.
    x_min, x_max = float(x.min()), float(x.max())
    span = (x_max - x_min) if (x_max - x_min) > 1e-12 else 1.0
    x0 = 0.5 * (x_min + x_max)
    xs = (x - x0) / span  # centered, span-normalized training inputs

    # Standardize targets (GP signal variance will absorb scale).
    y_mean = float(y.mean())
    y_std = float(y.std()) if y.std() > 1e-12 else 1.0
    ys = (y - y_mean) / y_std

    noise = 1e-8  # near-zero noise: data is noiseless -> exact interpolation

    def kernel(lg_l, lg_s2):
        ell = float(np.exp(lg_l))
        s2 = float(np.exp(lg_s2))
        d = xs[:, None] - xs[None, :]
        K = s2 * np.exp(-0.5 * (d * d) / (ell * ell))
        K[np.diag_indices_from(K)] += noise
        return K

    def logmll(params):
        lg_l, lg_s2 = params
        ell = float(np.exp(lg_l))
        s2 = float(np.exp(lg_s2))
        if not (np.isfinite(ell) and np.isfinite(s2)) or ell <= 0 or s2 <= 0:
            return -1e12
        d = xs[:, None] - xs[None, :]
        K = s2 * np.exp(-0.5 * (d * d) / (ell * ell))
        K[np.diag_indices_from(K)] += noise
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return -1e12
        # Solve L a = ys
        a = np.linalg.solve(L, ys)
        alpha = np.linalg.solve(L.T, a)
        mll = -0.5 * float(ys @ alpha) - float(np.sum(np.log(np.diag(L))))
        if not np.isfinite(mll):
            return -1e12
        return mll

    def neg_logmll(params):
        return -logmll(params)

    best = None
    best_score = -np.inf

    # Multi-start over length-scale (log-uniform across plausible scales) and signal variance.
    starts = []
    for lg_l in np.linspace(-3.0, 2.0, 13):
        for lg_s2 in np.linspace(-1.5, 1.5, 5):
            starts.append((lg_l, lg_s2))

    for (lg_l, lg_s2) in starts:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                neg_logmll,
                np.array([lg_l, lg_s2], dtype=float),
                method="L-BFGS-B",
                bounds=[(-4.0, 4.0), (-3.0, 3.0)],
                options={"maxiter": 60, "ftol": 1e-9, "gtol": 1e-8},
            )
        except Exception:
            continue
        score = -float(res.fun) if np.isfinite(res.fun) else -np.inf
        if score > best_score:
            best_score = score
            best = np.asarray(res.x, dtype=float)

    if best is None:
        # GP failed everywhere: fall back to polynomial.
        return _poly_predict(X, y)

    lg_l, lg_s2 = float(best[0]), float(best[1])
    ell = float(np.exp(lg_l))
    s2 = float(np.exp(lg_s2))
    ell = max(ell, 1e-9)
    s2 = max(s2, 1e-9)

    # Precompute Cholesky of training kernel for fast prediction.
    d = xs[:, None] - xs[None, :]
    K = s2 * np.exp(-0.5 * (d * d) / (ell * ell))
    K[np.diag_indices_from(K)] += noise
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        return _poly_predict(X, y)
    a = np.linalg.solve(L, ys)
    alpha = np.linalg.solve(L.T, a)  # K^{-1} y

    def predict(X2):
        X2 = np.asarray(X2, dtype=float)
        x2 = X2[:, 0].copy()
        xs2 = (x2 - x0) / span
        # k(x*, X) : (m, n)
        d2 = xs2[:, None] - xs[None, :]
        k = s2 * np.exp(-0.5 * (d2 * d2) / (ell * ell))
        # mean = k @ alpha, then denormalize target
        mu = k @ alpha
        out = mu * y_std + y_mean
        # Guard against non-finite values.
        out = np.where(np.isfinite(out), out, y_mean)
        return out

    return predict
# EVOLVE-BLOCK-END
