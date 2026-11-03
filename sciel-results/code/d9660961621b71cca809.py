import numpy as np
import time
from scipy.optimize import minimize

# ---- Task constants (interface) ----
N = 50
DELTA = 0.5 / N
_XC = -0.25 + (np.arange(N) + 0.5) * DELTA

_DEADLINE = 145.0
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _ratio(h):
    """Raw objective: L2^2 / (L1 * max_autoconv)."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < 0):
        return -np.inf
    l1 = np.sum(h)
    if l1 <= 0.0:
        return 0.0
    l2_sq = np.dot(h, h)
    if l2_sq <= 0.0:
        return 0.0
    conv = np.convolve(h, h, mode="full")
    m = np.max(conv)
    if m <= 0.0:
        return 0.0
    return l2_sq / (l1 * m)

def _gaussian_h(params):
    """Construct h from Gaussian parameters: [A, center, sigma, A2, center2, sigma2]"""
    A, c, s, A2, c2, s2 = params
    x = np.arange(N, dtype=np.float64)
    h = A * np.exp(-(x - c)**2 / (2 * s**2))
    if len(params) >= 6:
        h += A2 * np.exp(-(x - c2)**2 / (2 * s2**2))
    return np.maximum(h, 0.0)

def _neg_ratio(params):
    h = _gaussian_h(params)
    r = _ratio(h)
    if r <= 0:
        return 1e10
    return -r

def construct():
    global _START
    _START = time.time()
    
    best_h = None
    best_r = 0.0
    
    # Try various initial guesses for single and double Gaussians
    x = np.arange(N, dtype=np.float64)
    
    # Single Gaussian search
    for center in [20, 22, 23, 24, 25, 26, 27, 28, 30]:
        for sigma in [2, 3, 4, 5, 6, 8, 10, 12]:
            if _time_left() < 10:
                break
            h = np.exp(-(x - center)**2 / (2 * sigma**2))
            r = _ratio(h)
            if r > best_r:
                best_r = r
                best_h = h.copy()
        
        # Local optimization from this starting point
        if _time_left() < 15:
            continue
        try:
            result = minimize(_neg_ratio, [1.0, center, sigma, 0.0, center, sigma],
                            method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.success or -result.fun > best_r:
                h_opt = _gaussian_h(result.x)
                r_opt = _ratio(h_opt)
                if r_opt > best_r:
                    best_r = r_opt
                    best_h = h_opt.copy()
        except:
            pass
    
    # Double Gaussian search
    for c1 in [15, 20, 25]:
        for c2 in [30, 35, 40]:
            for s in [3, 5, 8]:
                if _time_left() < 10:
                    break
                h = np.exp(-(x - c1)**2 / (2 * s**2)) + np.exp(-(x - c2)**2 / (2 * s**2))
                r = _ratio(h)
                if r > best_r:
                    best_r = r
                    best_h = h.copy()
                
                if _time_left() < 15:
                    continue
                try:
                    x0 = [1.0, c1, s, 1.0, c2, s]
                    result = minimize(_neg_ratio, x0,
                                    method='Nelder-Mead',
                                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
                    if result.success or -result.fun > best_r:
                        h_opt = _gaussian_h(result.x)
                        r_opt = _ratio(h_opt)
                        if r_opt > best_r:
                            best_r = r_opt
                            best_h = h_opt.copy()
                except:
                    pass
    
    # Also try uniform and triangular as baselines
    for h_test in [np.ones(N), np.abs(x - 25) + 1, np.sin(np.pi * x / N)**2]:
        r = _ratio(h_test)
        if r > best_r:
            best_r = r
            best_h = h_test.copy()
    
    if best_h is None:
        best_h = np.zeros(N)
        best_h[0] = 1.0
    
    return list(best_h)
