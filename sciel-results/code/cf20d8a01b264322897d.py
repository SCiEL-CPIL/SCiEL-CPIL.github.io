import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    """Compute the raw score: L2^2 / (L1 * AutoConvMax)."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def construct():
    # We want to maximize l2_sq / (l1 * autoconv_max)
    # Fix l1 = 1. Maximize l2_sq / autoconv_max.
    # Minimize autoconv_max / l2_sq.
    # Or simply maximize the ratio directly.
    
    def objective(h):
        l1 = np.sum(h)
        if l1 < 1e-12:
            return 0.0
        l2_sq = np.sum(h**2)
        h_conv = np.convolve(h, h, mode='full')
        autoconv_max = DELTA * np.max(h_conv)
        if autoconv_max < 1e-12:
            return 0.0
        return - (l2_sq / (l1 * autoconv_max))

    # Constraints
    # sum(h) = 1
    cons = [{'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0}]
    bounds = [(0, None) for _ in range(N)]
    
    best_h = None
    best_score = -np.inf
    
    # Try several random initializations
    # 1. Uniform
    h0 = np.ones(N) / N
    res = minimize(objective, h0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 500, 'ftol': 1e-12})
    if res.success:
        h_res = np.maximum(res.x, 0)
        h_res = h_res / np.sum(h_res) # Renormalize
        score = -objective(h_res)
        if score > best_score:
            best_score = score
            best_h = h_res
            
    # 2. Sparse random starts (mimicking Sidon sets)
    for _ in range(20):
        k = np.random.randint(3, 12) # Try small sparse sets
        indices = np.random.choice(N, k, replace=False)
        h0 = np.zeros(N)
        h0[indices] = 1.0 / k
        res = minimize(objective, h0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 500, 'ftol': 1e-12})
        if res.success:
            h_res = np.maximum(res.x, 0)
            h_res = h_res / np.sum(h_res)
            score = -objective(h_res)
            if score > best_score:
                best_score = score
                best_h = h_res
                
    if best_h is None:
        # Fallback to uniform
        best_h = np.ones(N) / N
        
    return list(best_h)
