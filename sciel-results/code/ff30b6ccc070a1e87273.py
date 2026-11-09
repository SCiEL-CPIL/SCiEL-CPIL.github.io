# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

_DEADLINE = 600.0
_START = time.time()


def _remaining() -> float:
    return _DEADLINE - (time.time() - _START)


def _compute_P(params: np.ndarray) -> np.ndarray:
    """Build P[y, x] from (alpha, beta, gamma): weight = exp(-a*x^2 - b*y^2 - g*x*y)."""
    a, b, g = params
    n = 10
    x = np.arange(n, dtype=np.float64)
    y = np.arange(n, dtype=np.float64)
    X, Y = np.meshgrid(x, y, indexing='xy')
    W = np.exp(-a * X * X - b * Y * Y - g * X * Y)
    W = np.clip(W, 0.0, None)
    s = W.sum()
    if s <= 1e-300:
        return np.ones((n, n), dtype=np.float64) / (n * n)
    return W / s


def _entropy_of_projection(P: np.ndarray, r: float) -> float:
    """Shannon entropy (bits) of the distribution of X + r*Y."""
    n = P.shape[0]
    x = np.arange(n, dtype=np.float64)
    y = np.arange(n, dtype=np.float64)
    X, Y = np.meshgrid(x, y, indexing='xy')
    vals = X + r * Y
    probs = np.zeros(n * n, dtype=np.float64)
    for i in range(n * n):
        v = vals.flat[i]
        probs[i] = P.flat[i]
    # pool by value
    uniq, inv = np.unique(vals, return_inverse=True)
    pooled = np.zeros(len(uniq), dtype=np.float64)
    np.add.at(pooled, inv, probs)
    p = pooled[pooled > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())


def _objective(params: np.ndarray) -> float:
    """Negative ratio D/M (we minimize)."""
    P = _compute_P(params)
    D = _entropy_of_projection(P, -1.0)
    H0 = _entropy_of_projection(P, 0.0)
    H1 = _entropy_of_projection(P, 1.0)
    H2 = _entropy_of_projection(P, 2.0)
    M = max(H0, H1, H2)
    if M <= 1e-9:
        return 1e9
    return -(D / M)


def _uniform_ratio() -> float:
    n = 10
    P = np.ones((n, n), dtype=np.float64) / (n * n)
    D = _entropy_of_projection(P, -1.0)
    M = max(_entropy_of_projection(P, r) for r in (0.0, 1.0, 2.0))
    return D / M


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.

    Uses a 3-parameter product ansatz P[x,y] ∝ exp(-α x² - β y² - γ xy),
    optimized with differential evolution followed by a local polish.
    """
    best_P = np.ones((10, 10), dtype=np.float64) / 100.0
    best_score = _uniform_ratio()

    # Global search with differential evolution
    bounds = [(-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0)]
    try:
        if _remaining() > 60:
            res = differential_evolution(
                _objective,
                bounds,
                maxiter=80,
                popsize=15,
                tol=1e-8,
                seed=42,
                polish=False,
                init='sobol',
                updating='deferred',
            )
            if _remaining() > 30:
                cand_P = _compute_P(res.x)
                cand_score = -_objective(res.x)
                if cand_score > best_score:
                    best_score = cand_score
                    best_P = cand_P
    except Exception:
        pass

    # Local polish from the best found (and a few extra starts)
    starts = []
    try:
        starts.append(res.x)
    except Exception:
        pass
    # Add a couple of hand-picked starts
    starts.append(np.array([0.0, 0.0, 0.0]))
    starts.append(np.array([0.3, 0.3, -0.5]))
    starts.append(np.array([0.5, 0.5, 0.0]))
    starts.append(np.array([0.0, 0.0, 0.8]))
    starts.append(np.array([0.2, 0.2, 0.6]))

    for s in starts:
        if _remaining() < 10:
            break
        try:
            r2 = minimize(_objective, s, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 200, 'ftol': 1e-12})
            if r2.success or r2.fun < _objective(s):
                cand_P = _compute_P(r2.x)
                cand_score = -r2.fun
                if cand_score > best_score:
                    best_score = cand_score
                    best_P = cand_P
        except Exception:
            pass

    # Final safety: clip and normalize
    best_P = np.clip(best_P, 0.0, None)
    s = best_P.sum()
    if s <= 1e-9:
        best_P = np.ones((10, 10), dtype=np.float64) / 100.0
    else:
        best_P = best_P / s
    return best_P.astype(np.float64)


# EVOLVE-BLOCK-END
