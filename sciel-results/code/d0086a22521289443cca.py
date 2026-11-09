# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] for discrete measure with given support and weights."""
    w = np.asarray(weights, dtype=np.float64)
    s = np.asarray(support, dtype=np.float64)
    total = w.sum()
    if total <= 1e-9:
        return 0.0
    w = w / total
    n = len(s)
    # Build pairwise sums s_i + s_j
    ss = s[:, None] + s[None, :]
    # P = sum_l w_l * sum_{i,j,k: s_i+s_j+s_k < 2*s_l} w_i w_j w_k
    # For each l, compute triple sum
    P = 0.0
    for l in range(n):
        target = 2.0 * s[l]
        # For each pair (i,j), compute sum over k with s_k < target - s_i - s_j
        # More efficient: for each (i,j), find k such that s_k < target - s_i - s_j
        # Precompute: for each (i,j), the threshold for k is target - s_i - s_j
        # triple = sum_{i,j} w_i w_j * sum_{k: s_k < target - s_i - s_j} w_k
        # We can precompute for each threshold, the cumulative weight
        # But thresholds vary per (i,j)
        # For small n, just do it directly
        triple = 0.0
        for i in range(n):
            for j in range(n):
                thresh = target - s[i] - s[j]
                # sum w_k for k with s_k < thresh
                # Since s is sorted, find index
                idx = np.searchsorted(s, thresh, side='left')
                if idx > 0:
                    triple += w[i] * w[j] * w[:idx].sum()
        P += w[l] * triple
    return P

def _neg_P(params, support):
    """Negative P for minimization. params are weights (unnormalized, non-negative)."""
    w = np.exp(params)  # ensure positive
    return -_compute_P(support, w)

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 580  # 20s safety margin
    
    best_score = 0.0
    best_support = None
    best_weights = None
    
    # Candidate 1: 2-point optimal
    # g(p) = 3p^2 - 5p^3 + 2p^4, max at p = (15 - sqrt(33))/16
    p_opt = (15.0 - np.sqrt(33.0)) / 16.0
    # M can be any large value; use M=1 for simplicity (scale invariance? No, support must be fixed)
    # Actually, the value of M doesn't matter for the 2-point case as long as M > 0
    s2 = np.array([0.0, 1.0])
    w2 = np.array([p_opt, 1.0 - p_opt])
    P2 = _compute_P(s2, w2)
    if P2 > best_score:
        best_score = P2
        best_support = s2.copy()
        best_weights = w2.copy()
    
    if time.time() > deadline:
        return best_support, best_weights
    
    # Candidate 2: 3-point optimization
    # Support: {0, a, 1} with a in (0,1), optimize weights
    for a_val in [0.25, 0.5, 0.75]:
        s3 = np.array([0.0, a_val, 1.0])
        # Initial weights
        w0 = np.array([0.3, 0.3, 0.4])
        x0 = np.log(w0)
        try:
            res = minimize(_neg_P, x0, args=(s3,), method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            w3 = np.exp(res.x)
            P3 = -res.fun
            if P3 > best_score:
                best_score = P3
                best_support = s3.copy()
                best_weights = w3.copy()
        except Exception:
            pass
        if time.time() > deadline:
            break
    
    # Candidate 3: 4-point optimization
    s4 = np.array([0.0, 0.33, 0.66, 1.0])
    w0 = np.ones(4) / 4
    x0 = np.log(w0)
    try:
        res = minimize(_neg_P, x0, args=(s4,), method='Nelder-Mead',
                      options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10})
        w4 = np.exp(res.x)
        P4 = -res.fun
        if P4 > best_score:
            best_score = P4
            best_support = s4.copy()
            best_weights = w4.copy()
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_support, best_weights
    
    # Candidate 4: 5-point optimization
    s5 = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    w0 = np.ones(5) / 5
    x0 = np.log(w0)
    try:
        res = minimize(_neg_P, x0, args=(s5,), method='Nelder-Mead',
                      options={'maxiter': 4000, 'xatol': 1e-8, 'fatol': 1e-10})
        w5 = np.exp(res.x)
        P5 = -res.fun
        if P5 > best_score:
            best_score = P5
            best_support = s5.copy()
            best_weights = w5.copy()
    except Exception:
        pass
    
    if best_support is None:
        # Fallback to uniform
        n = 32
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n
    
    return best_support, best_weights

# EVOLVE-BLOCK-END
