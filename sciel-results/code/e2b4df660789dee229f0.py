import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def calculate_P(support, weights):
    """
    Calculate P[X1 + X2 + X3 < 2*X4]
    """
    n = len(support)
    if n == 0:
        return 0.0
    w = weights / np.sum(weights)
    
    # Precompute pairwise sums for X1+X2 (size n x n)
    S_12 = support[:, None] + support[None, :]
    W_12 = w[:, None] * w[None, :]
    
    P_total = 0.0
    for k in range(n):
        target = 2 * support[k]
        prob_k = 0.0
        for l in range(n):
            thresh = target - support[l]
            prob_l = np.sum(W_12[S_12 < thresh])
            prob_k += w[l] * prob_l
        P_total += w[k] * prob_k
        
    return P_total

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = 580.0
    
    best_P = -1.0
    best_support = None
    best_weights = None
    
    def update_best(s, w):
        nonlocal best_P, best_support, best_weights
        if s is None or w is None:
            return
        try:
            if len(s) != len(w): return
            if len(s) == 0: return
            if np.any(np.isnan(s)) or np.any(np.isnan(w)): return
            if np.any(np.isinf(s)) or np.any(np.isinf(w)): return
            if np.any(s < -1e-9) or np.any(w < -1e-9): return
            if np.sum(w) <= 1e-9: return
            
            p_val = calculate_P(s, w)
            if p_val > best_P:
                best_P = p_val
                best_support = s.copy()
                best_weights = w.copy()
        except Exception:
            pass

    # 1. Analytical 2-point solution
    # Maximize p(1-p)^2(1+2p)
    # Optimal p ~ 0.41
    p_opt = 0.41
    s_2pt = np.array([0.0, 10.0])
    w_2pt = np.array([1-p_opt, p_opt])
    update_best(s_2pt, w_2pt)
    
    # 2. Differential Evolution for small n
    # We optimize parameters: supports (n) and weights (n)
    # Constraints: s >= 0, w >= 0, sum(w) > 0
    
    def obj_de(params):
        n = len(params) // 2
        s = params[:n]
        w = params[n:]
        
        # Clip to valid ranges
        s = np.clip(s, 0, 100)
        w = np.clip(w, 0, 1)
        
        w_sum = np.sum(w)
        if w_sum < 1e-9:
            return 0.0
        w = w / w_sum
        
        # Check validity quickly
        if np.any(np.isnan(s)) or np.any(np.isnan(w)):
            return 0.0
            
        p_val = calculate_P(s, w)
        return -p_val

    rng = np.random.default_rng(42)
    
    for n_pts in [3, 4, 5, 6]:
        if time.time() > deadline - 60:
            break
            
        # Bounds for DE
        bounds = [(0, 50)] * n_pts + [(0, 1)] * n_pts
        
        # Initialize DE
        try:
            res = differential_evolution(
                obj_de, 
                bounds, 
                seed=42, 
                maxiter=100, 
                popsize=10, 
                tol=1e-4,
                polish=True # Use local minimization on the best result
            )
            
            if res.success:
                params_opt = res.x
                s_opt = params_opt[:n_pts]
                w_opt = params_opt[n_pts:]
                update_best(s_opt, w_opt)
                
                # Refine with Nelder-Mead or SLSQP if time permits
                if time.time() < deadline - 30:
                    def obj_refine(params):
                        s = params[:n_pts]
                        w = params[n_pts:]
                        s = np.clip(s, 0, 100)
                        w = np.clip(w, 0, 1)
                        w_sum = np.sum(w)
                        if w_sum < 1e-9: return 0.0
                        w = w / w_sum
                        return -calculate_P(s, w)
                    
                    # Use L-BFGS-B or Nelder-Mead
                    # Nelder-Mead is derivative free and robust for small n
                    res_ref = minimize(obj_refine, params_opt, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
                    if res_ref.success:
                        s_ref = res_ref.x[:n_pts]
                        w_ref = res_ref.x[n_pts:]
                        update_best(s_ref, w_ref)
                        
        except Exception:
            pass

    # Fallback
    if best_support is None:
        best_support = np.arange(32, dtype=np.float64)
        best_weights = np.ones(32, dtype=np.float64) / 32
        
    return best_support, best_weights
