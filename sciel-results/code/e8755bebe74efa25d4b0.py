import numpy as np
from scipy.optimize import minimize
import time

N = 50
_START_TIME = time.time()
TIME_LIMIT = 110  # safety margin before 120s

def _score(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    l2_sq = np.sum(h * h)
    # linear autoconvolution
    conv = np.convolve(h, h, mode='full')
    sup = np.max(np.abs(conv))
    if sup <= 0:
        return 0.0
    return l2_sq / (l1 * sup)

def _neg_score(h):
    return -_score(h)

def _project_nonneg(x):
    return np.maximum(x, 0.0)

def construct():
    global _START_TIME
    if time.time() - _START_TIME > TIME_LIMIT:
        return [1.0] * N
    
    best_h = np.ones(N)
    best_val = _score(best_h)
    
    # Try a spike
    spike = np.zeros(N)
    spike[N // 2] = 1.0
    v = _score(spike)
    if v > best_val:
        best_val = v
        best_h = spike.copy()
    
    # Try various structured initializations and optimize
    rng = np.random.default_rng(42)
    
    # Generate diverse starting points
    starts = []
    # Spike
    starts.append(spike.copy())
    # Two spikes
    s2 = np.zeros(N)
    s2[10] = 1.0
    s2[40] = 1.0
    starts.append(s2)
    # Gaussian bump
    x = np.linspace(-1, 1, N)
    g = np.exp(-5 * x**2)
    starts.append(g / np.sum(g) * np.sum(g))  # keep as is
    # Random
    for _ in range(5):
        r = rng.random(N)
        starts.append(r)
    # Ramp
    rmp = np.linspace(0, 1, N)
    starts.append(rmp)
    # Symmetric ramp up then down
    sym = np.concatenate([np.linspace(0, 1, N//2), np.linspace(1, 0, N - N//2)])
    starts.append(sym)
    
    for h0 in starts:
        if time.time() - _START_TIME > TIME_LIMIT:
            break
        h0 = np.asarray(h0, dtype=np.float64)
        h0 = _project_nonneg(h0)
        if np.sum(h0) <= 0:
            continue
        
        # Use L-BFGS-B to maximize score (minimize negative)
        try:
            res = minimize(_neg_score, h0, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            h_opt = _project_nonneg(res.x)
            if np.sum(h_opt) > 0:
                v = _score(h_opt)
                if v > best_val:
                    best_val = v
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Polish: use the best found as starting point for more iterations
    if time.time() - _START_TIME < TIME_LIMIT - 10:
        try:
            res = minimize(_neg_score, best_h, method='L-BFGS-B',
                          options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12})
            h_pol = _project_nonneg(res.x)
            if np.sum(h_pol) > 0:
                v = _score(h_pol)
                if v > best_val:
                    best_val = v
                    best_h = h_pol.copy()
        except Exception:
            pass
    
    # Ensure nonnegativity and return as list of floats
    best_h = np.maximum(best_h, 0.0)
    return [float(x) for x in best_h]
