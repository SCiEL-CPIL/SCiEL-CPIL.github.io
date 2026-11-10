# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _entropy_of_projection(P, r):
    """Shannon entropy (bits) of the projected distribution x + r*y."""
    n = P.shape[0]
    xs = np.arange(n, dtype=np.float64)
    ys = np.arange(n, dtype=np.float64)
    vals = xs[None, :] + r * ys[:, None]
    uniq, inv = np.unique(vals, return_inverse=True)
    pooled = np.bincount(inv.ravel(), weights=P.ravel(), minlength=len(uniq))
    pooled = pooled[pooled > 0.0]
    return float(-(pooled * np.log2(pooled)).sum())

def _ratio(P):
    D = _entropy_of_projection(P, -1.0)
    M = max(_entropy_of_projection(P, 0.0),
            _entropy_of_projection(P, 1.0),
            _entropy_of_projection(P, 2.0))
    if M <= 1e-9:
        return -np.inf
    return D / M

def _objective(log_w):
    w = log_w - np.max(log_w)
    p = np.exp(w)
    s = p.sum()
    if s < 1e-300:
        return 1e10
    P = p / s
    P = P.reshape(10, 10)
    r = _ratio(P)
    if not np.isfinite(r):
        return 1e10
    return -r

def _optimize_from_start(P_init):
    log_w_init = np.log(np.clip(P_init.ravel(), 1e-300, None))
    try:
        res = minimize(_objective, log_w_init, method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-12, 'maxls': 50})
        log_w_opt = res.x
        w = log_w_opt - np.max(log_w_opt)
        p = np.exp(w)
        P_opt = p / p.sum()
        P_opt = P_opt.reshape(10, 10)
        return P_opt
    except Exception:
        return P_init

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    best_P = None
    best_score = -np.inf
    
    # Strategy 1: Uniform + Noise
    np.random.seed(1)
    P1 = np.ones((n, n)) + np.random.normal(0, 0.5, (n, n))
    P1 = np.clip(P1, 1e-6, None)
    P1 = P1 / P1.sum()
    P_opt1 = _optimize_from_start(P1)
    score1 = _ratio(P_opt1)
    if score1 > best_score:
        best_score = score1
        best_P = P_opt1

    # Strategy 2: Two Blocks (from previous heuristic, but optimized)
    P2 = np.zeros((n, n))
    P2[0:5, 0:5] = 1.0
    P2[5:10, 5:10] = 1.0
    P2 = P2 / P2.sum()
    P_opt2 = _optimize_from_start(P2)
    score2 = _ratio(P_opt2)
    if score2 > best_score:
        best_score = score2
        best_P = P_opt2

    # Strategy 3: Diagonals
    P3 = np.zeros((n, n))
    for i in range(n):
        P3[i, i] = 1.0
        P3[i, (i + 5) % n] = 1.0
    P3 = P3 / P3.sum()
    P_opt3 = _optimize_from_start(P3)
    score3 = _ratio(P_opt3)
    if score3 > best_score:
        best_score = score3
        best_P = P_opt3

    # Final safety
    P = np.clip(best_P, 0.0, None)
    s = P.sum()
    if s <= 1e-9:
        P = np.ones((n, n)) / (n * n)
    else:
        P = P / s
        
    return P
# EVOLVE-BLOCK-END
