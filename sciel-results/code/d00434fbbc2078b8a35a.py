import numpy as np
from scipy.optimize import minimize

N = 400

def _evaluate_h(h):
    S = np.sum(h)
    if np.abs(S) < 1e-12:
        return np.inf
    C = np.convolve(h, h)
    M = np.max(C)
    return abs(2 * N * M / (S * S))

def _construct_h(params):
    """
    params: [a0, w1, a1, b1, w2, a2, b2, ...]
    h_i = a0 + sum( a_j sin(w_j i) + b_j cos(w_j i) )
    """
    i = np.arange(N)
    h = np.full(N, params[0])
    idx = 1
    while idx < len(params):
        w = params[idx]
        a = params[idx+1]
        b = params[idx+2]
        h += a * np.sin(w * i) + b * np.cos(w * i)
        idx += 3
    return h

def construct():
    best_h = np.ones(N)
    best_val = _evaluate_h(best_h)
    
    # Try different numbers of harmonics
    for K in [1, 2, 3, 4]:
        # Number of parameters: 1 (DC) + 3*K
        n_params = 1 + 3 * K
        
        # Random restarts
        for _ in range(200):
            # Initialize parameters
            # DC term
            a0 = np.random.uniform(-1, 1)
            params = np.array([a0])
            
            for j in range(K):
                w = np.random.uniform(0, 2 * np.pi)
                a = np.random.uniform(-1, 1)
                b = np.random.uniform(-1, 1)
                params = np.append(params, [w, a, b])
            
            # Local optimization using Nelder-Mead
            # We optimize the value directly
            # Note: Nelder-Mead might struggle with discontinuities in phase (sin/cos periodicity)
            # but random restarts help.
            
            def objective(p):
                h = _construct_h(p)
                return _evaluate_h(h)
            
            try:
                res = minimize(objective, params, method='Nelder-Mead', 
                               options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
                if res.fun < best_val:
                    best_val = res.fun
                    best_h = _construct_h(res.x)
            except Exception:
                pass
                
    return best_h
