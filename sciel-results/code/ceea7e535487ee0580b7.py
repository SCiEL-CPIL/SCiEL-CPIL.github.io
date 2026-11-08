import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)  # enforce nonnegativity
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h ** 2)
    # Linear autoconvolution
    hh = np.convolve(h, h, mode='full')
    hh_inf = np.max(np.abs(hh))
    if hh_inf < 1e-15:
        return 0.0
    return l2_sq / (l1 * hh_inf)

def _negative_ratio(h):
    return -_ratio(h)

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    # Try several initial configurations
    inits = [
        np.ones(N),
        np.zeros(N),
    ]
    # 2-point clusters
    for i in range(N - 1):
        for ratio_val in [0.5, 1.0, 2.0, 0.3, 3.0]:
            h = np.zeros(N)
            h[i] = 1.0
            h[i+1] = ratio_val
            inits.append(h)
    # 3-point clusters
    for i in range(N - 2):
        for a, b, c in [(1, 0.5, 0.25), (1, 1, 0.5), (2, 1, 0.5), (1, 0.7, 0.4)]:
            h = np.zeros(N)
            h[i] = a
            h[i+1] = b
            h[i+2] = c
            inits.append(h)
    # Sparse single point
    for i in range(N):
        h = np.zeros(N)
        h[i] = 1.0
        inits.append(h)
    
    # Evaluate all inits
    for h in inits:
        if time.time() > deadline - 5:
            break
        s = _ratio(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    # Local optimization from best init
    if time.time() < deadline - 10:
        try:
            # Use L-BFGS-B with nonnegativity constraints
            bounds = [(0.0, None)] * N
            res = minimize(_negative_ratio, best_h, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.success or res.fun < -best_score:
                h_opt = np.maximum(res.x, 0.0)
                s_opt = _ratio(h_opt)
                if s_opt > best_score:
                    best_score = s_opt
                    best_h = h_opt
        except Exception:
            pass
    
    # Try more random restarts if time permits
    rng = np.random.default_rng(42)
    attempts = 0
    while time.time() < deadline - 5 and attempts < 200:
        attempts += 1
        # Random sparse vectors
        k = rng.integers(1, min(10, N))
        idx = rng.choice(N, size=k, replace=False)
        h = np.zeros(N)
        h[idx] = rng.uniform(0.1, 2.0, size=k)
        s = _ratio(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
        # Local opt from random
        if time.time() < deadline - 5:
            try:
                bounds = [(0.0, None)] * N
                res = minimize(_negative_ratio, h, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
                h_opt = np.maximum(res.x, 0.0)
                s_opt = _ratio(h_opt)
                if s_opt > best_score:
                    best_score = s_opt
                    best_h = h_opt
            except Exception:
                pass
    
    return list(best_h)
# EVOLVE-BLOCK-END
