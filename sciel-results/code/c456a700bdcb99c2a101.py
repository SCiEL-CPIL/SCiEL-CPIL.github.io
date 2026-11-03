import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    """
    Compute the score: L2^2 / (L1 * Sup(autoconv))
    Assumes h is the vector of heights.
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max)) / (DELTA**2)

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    best_h = np.zeros(N)
    best_h[0] = 1.0
    best_score = _score_raw(best_h)
    
    # Objective to maximize: sum(h^2) / max(conv(h,h))
    # Subject to sum(h)=1, h>=0
    
    def neg_obj(h):
        l1 = np.sum(h)
        if l1 < 1e-12: return 0
        l2_sq = np.sum(h**2)
        hc = np.convolve(h, h, mode='full')
        mc = np.max(hc)
        if mc < 1e-12: return 0
        return -(l2_sq / mc)
        
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0},
    ]
    
    bounds = [(0.0, 1.0)] * N
    
    # Strategy: Try diverse initializations and refine.
    # 1. Delta
    inits = [best_h.copy()]
    
    # 2. Uniform
    h_uni = np.ones(N) / N
    inits.append(h_uni)
    
    # 3. Random Sparse (Sidon-like attempts)
    for _ in range(100):
        k = np.random.randint(2, 10)
        idx = np.random.choice(N, k, replace=False)
        h_s = np.zeros(N)
        h_s[idx] = 1.0
        h_s /= np.sum(h_s)
        inits.append(h_s)
        
    # 4. Random Dense
    for _ in range(100):
        h_r = np.random.rand(N)
        h_r /= np.sum(h_r)
        inits.append(h_r)
        
    for h_init in inits:
        if time.time() > deadline: break
        try:
            res = minimize(neg_obj, h_init, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 200, 'ftol': 1e-10})
            if res.success:
                h_candidate = res.x
                h_candidate = np.clip(h_candidate, 0, None)
                s = np.sum(h_candidate)
                if s > 0:
                    h_candidate /= s
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass
            
    # Return the best found
    return list(best_h)
