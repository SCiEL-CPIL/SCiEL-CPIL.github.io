import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _proxy_score(h):
    # h is a vector of non-negative numbers
    # Normalize to sum 1 to handle scale invariance
    s = np.sum(h)
    if s < 1e-9:
        return 0.0
    h = h / s
    
    # Discrete linear autoconvolution
    # Efficient way: use FFT or just np.convolve for small N
    c = np.convolve(h, h, mode='full')
    
    # L2 norm squared
    l2_sq = np.sum(h**2)
    
    # L1 norm
    l1 = np.sum(h)
    
    # Supremum norm of autoconvolution
    max_c = np.max(c)
    
    if max_c < 1e-9:
        return 0.0
        
    return (l2_sq / (l1 * max_c))

def construct():
    t0 = time.time()
    deadline = t0 + 100  # Safety margin
    
    # Initial guess: known best [2/3, 1/3, 0...]
    h0 = np.zeros(N)
    h0[0] = 2.0 / 3.0
    h0[1] = 1.0 / 3.0
    
    # Define bounds: h_i >= 0
    bounds = [(0.0, None)] * N
    
    # Objective: minimize negative proxy score
    def objective(h):
        return -_proxy_score(h)
    
    # Constraints: sum h = 1 (to fix scale)
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0}]
    
    best_h = h0.copy()
    best_score = _proxy_score(best_h)
    
    try:
        res = minimize(
            objective, 
            h0, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints, 
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        if res.success:
            h_opt = res.x
            h_opt[h_opt < 0] = 0 # Clean up small negatives
            h_opt = h_opt / np.sum(h_opt) # Renormalize
            score_opt = _proxy_score(h_opt)
            if score_opt > best_score:
                best_h = h_opt
                best_score = score_opt
    except Exception as e:
        pass
        
    if time.time() > deadline:
        return [float(x) for x in best_h]
        
    return [float(x) for x in best_h]
