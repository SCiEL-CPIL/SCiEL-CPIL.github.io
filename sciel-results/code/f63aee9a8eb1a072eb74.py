import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_ratio(h):
    h = np.asarray(h)
    if np.any(h < 0):
        return -np.inf
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 == 0:
        return 0
    # Linear autoconvolution
    autoconv = np.convolve(h, h, mode='full')
    sup_norm = np.max(np.abs(autoconv))
    if sup_norm == 0:
        return 0
    return l2_sq / (l1 * sup_norm)

def neg_ratio(params):
    h = np.exp(params) # Ensure positivity
    h = h / np.sum(h) # Normalize to L1=1 for shape optimization
    return -compute_ratio(h)

def construct():
    start_time = time.time()
    best_h = np.ones(N)
    best_score = compute_ratio(best_h)
    
    # Try to find a shape that maximizes the ratio
    # Since for non-negative h, ratio is 1/L1 (if L1=1, ratio=1), 
    # we try to see if any numerical artifact or specific shape helps.
    
    # Random restarts
    for _ in range(20):
        if time.time() - start_time > 100:
            break
        # Start with random positive shape
        h_init = np.random.rand(N)
        # Optimize shape
        try:
            res = minimize(neg_ratio, np.log(h_init), method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-4})
            h_opt = np.exp(res.x)
            h_opt = h_opt / np.sum(h_opt)
            score = compute_ratio(h_opt)
            if score > best_score:
                best_score = score
                best_h = h_opt
        except Exception:
            continue
            
    # Scale the best shape to maximize the ratio
    # If the ratio is scale dependent (R ~ 1/c), we want c small.
    # But usually there is a constraint like max(h) <= 1 or ||h||_1 <= 1.
    # If max(h) <= 1 is the constraint:
    # We scale best_h so that max(best_h) = 1.
    if np.max(best_h) > 0:
        best_h_scaled = best_h / np.max(best_h)
    else:
        best_h_scaled = best_h
        
    return list(best_h_scaled)
