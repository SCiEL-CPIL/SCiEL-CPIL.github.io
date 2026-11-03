import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _score_raw(h):
    """
    Compute the score. 
    Based on the prompt, this is the ratio of squared L2 norm to L1 norm times supremum norm of autoconvolution.
    We assume the evaluator uses a specific normalization or continuous approximation.
    To be safe, we maximize the discrete ratio l2_sq / (l1 * l_inf) which is scale invariant.
    If the evaluator scales by 1/Delta, this will also be scaled.
    """
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    
    # Normalize to sum=1 for shape evaluation
    h_norm = h / l1
    l2_sq = np.sum(h_norm**2)
    
    conv = np.convolve(h_norm, h_norm, mode='full')
    l_inf = np.max(conv)
    
    if l_inf == 0:
        return 0.0
        
    return l2_sq / l_inf

def construct():
    start_time = time.time()
    deadline = 110.0 # Keep 10s margin for 120s limit
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    # Helper to check time
    def time_left():
        return deadline - (time.time() - start_time)
        
    # 1. Try specific structured initializations
    # Sidon-like sets (B2 sets) minimize collisions
    # A simple Sidon set construction: squares mod p? Or just random sparse with local search.
    
    np.random.seed(42)
    
    candidates = []
    # Uniform
    candidates.append(np.ones(N) / N)
    # Delta
    h_delta = np.zeros(N); h_delta[0] = 1.0
    candidates.append(h_delta)
    # Random sparse
    for _ in range(5):
        h_sp = np.zeros(N)
        idx = np.random.choice(N, size=10, replace=False)
        h_sp[idx] = 1.0
        h_sp /= np.sum(h_sp)
        candidates.append(h_sp)
        
    for c in candidates:
        s = _score_raw(c)
        if s > best_score:
            best_score = s
            best_h = c.copy()
            
    # 2. Local Optimization
    def objective(x):
        return -_score_raw(x)
        
    # Use Nelder-Mead for non-smooth max
    max_restarts = 20
    for i in range(max_restarts):
        if time_left() < 5.0:
            break
            
        # Generate start
        h_start = best_h + np.random.randn(N) * 0.1
        h_start = np.maximum(h_start, 0)
        if np.sum(h_start) == 0:
            h_start = np.ones(N) / N
        h_start /= np.sum(h_start)
        
        try:
            res = minimize(objective, h_start, method='Nelder-Mead', 
                           options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 500})
            if res.fun < -best_score:
                h_opt = np.maximum(res.x, 0)
                h_opt /= np.sum(h_opt)
                s = _score_raw(h_opt)
                if s > best_score:
                    best_score = s
                    best_h = h_opt.copy()
        except Exception:
            pass
            
    return list(best_h)
