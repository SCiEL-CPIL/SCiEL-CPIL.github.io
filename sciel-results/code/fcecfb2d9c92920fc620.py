import numpy as np
import time
from scipy.optimize import minimize

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
            mask = S_12 < thresh
            prob_l = np.sum(W_12 * mask)
            prob_k += w[l] * prob_l
        P_total += w[k] * prob_k
        
    return P_total

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = 580.0
    
    best_P = -1.0
    best_support = None
    best_weights = None
    
    # Analytical two-point solution: support {0, L}, weights {1-p, p}
    # P = p*(1 - 3p^2 + 2p^3), maximize over p in (0,1)
    # f'(p) = 1 - 9p^2 + 8p^3 = 0
    # Try p=0.42: 1 - 9*0.1764 + 8*0.074088 = 1 - 1.5876 + 0.5927 = 0.0051
    # Try p=0.43: 1 - 9*0.1849 + 8*0.079507 = 1 - 1.6641 + 0.6361 = -0.0280
    # Root near p=0.422
    p_opt = 0.422
    L_opt = 10.0
    s_two = np.array([0.0, L_opt])
    w_two = np.array([1 - p_opt, p_opt])
    
    def update_best(s, w):
        nonlocal best_P, best_support, best_weights
        if s is None or w is None:
            return
        try:
            s = np.asarray(s, dtype=np.float64).flatten()
            w = np.asarray(w, dtype=np.float64).flatten()
            if len(s) != len(w) or len(s) == 0:
                return
            if np.any(np.isnan(s)) or np.any(np.isinf(s)):
                return
            if np.any(np.isnan(w)) or np.any(np.isinf(w)):
                return
            if np.any(s < -1e-9) or np.any(w < -1e-9):
                return
            if np.sum(w) <= 1e-9:
                return
            
            p_val = calculate_P(s, w)
            if p_val > best_P:
                best_P = p_val
                best_support = s.copy()
                best_weights = w.copy()
        except Exception:
            pass

    # 0. Evaluate the analytical two-point solution
    update_best(s_two, w_two)
    
    # 1. Try the known good uniform grid with weight optimization
    # This captures the "structural" improvement of non-uniform weights on fixed supports
    for n_grid in [16, 32, 64]:
        s_init = np.arange(n_grid, dtype=np.float64)
        w_init = np.ones(n_grid) / n_grid
        
        def objective(w_flat, s_fixed=s_init):
            w = np.clip(w_flat, 0, 1)
            s_sum = np.sum(w)
            if s_sum < 1e-9:
                return 0.0
            w = w / s_sum
            return -calculate_P(s_fixed, w)
        
        bounds = [(0, 1)] * n_grid
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        
        try:
            res = minimize(objective, w_init, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 100, 'ftol': 1e-6})
            if res.success:
                update_best(s_init, res.x)
        except Exception:
            pass
        
        if time.time() > deadline:
            return best_support, best_weights

    # 2. Try small support sizes with optimized positions AND weights
    # This allows the support points themselves to move, potentially finding a better structure
    def optimize_small_support(n_pts):
        if n_pts == 2:
            s_init = np.array([0.0, 10.0])
            w_init = np.array([1 - p_opt, p_opt])
        elif n_pts == 3:
            s_init = np.array([0.0, 1.0, 10.0])
            w_init = np.array([0.5, 0.2, 0.3])
        elif n_pts == 4:
            s_init = np.array([0.0, 0.5, 2.0, 10.0])
            w_init = np.array([0.5, 0.2, 0.1, 0.2])
        elif n_pts == 5:
            s_init = np.array([0.0, 0.5, 2.0, 10.0, 50.0])
            w_init = np.array([0.45, 0.15, 0.1, 0.2, 0.1])
        elif n_pts == 6:
            s_init = np.array([0.0, 0.3, 1.0, 3.0, 10.0, 40.0])
            w_init = np.array([0.4, 0.15, 0.1, 0.1, 0.15, 0.1])
        elif n_pts == 7:
            s_init = np.array([0.0, 0.2, 0.5, 1.5, 4.0, 12.0, 50.0])
            w_init = np.array([0.35, 0.12, 0.1, 0.1, 0.1, 0.15, 0.08])
        elif n_pts == 8:
            s_init = np.array([0.0, 0.15, 0.4, 1.0, 2.5, 6.0, 15.0, 60.0])
            w_init = np.array([0.32, 0.1, 0.08, 0.08, 0.08, 0.1, 0.12, 0.12])
        else:
            return
        
        def obj_small(params):
            s = params[:n_pts]
            w = params[n_pts:]
            
            s = np.clip(s, 0, 1e6)
            w = np.clip(w, 0, 1)
            w_sum = np.sum(w)
            if w_sum < 1e-9:
                return 0.0
            w = w / w_sum
            
            return -calculate_P(s, w)
        
        bounds_s = [(0, 1000)] * n_pts
        bounds_w = [(0, 1)] * n_pts
        bounds = bounds_s + bounds_w
        
        best_local_p = -1.0
        best_local_params = None
        
        rng = np.random.default_rng(42 + n_pts)
        
        n_restarts = 5 if n_pts <= 4 else 3
        for restart in range(n_restarts):
            s_pert = s_init * (1 + 0.1 * rng.standard_normal(n_pts))
            s_pert = np.clip(s_pert, 0, 1000)
            w_pert = w_init * (1 + 0.1 * rng.standard_normal(n_pts))
            w_pert = np.clip(w_pert, 0.01, 1)
            
            params_init = np.concatenate([s_pert, w_pert])
            
            try:
                res = minimize(obj_small, params_init, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4})
                if res.fun < best_local_p:
                    best_local_p = res.fun
                    best_local_params = res.x
            except Exception:
                pass
            
            if time.time() > deadline:
                break
        
        if best_local_params is not None:
            s_opt = best_local_params[:n_pts]
            w_opt = best_local_params[n_pts:]
            update_best(s_opt, w_opt)
        
        # Also try a refinement pass with L-BFGS-B on the best found
        if best_local_params is not None and time.time() < deadline - 10:
            s_best = best_local_params[:n_pts]
            w_best = best_local_params[n_pts:]
            
            def obj_refine(params):
                s = params[:n_pts]
                w = params[n_pts:]
                s = np.clip(s, 0, 1e6)
                w = np.clip(w, 0, 1)
                w_sum = np.sum(w)
                if w_sum < 1e-9:
                    return 0.0
                w = w / w_sum
                return -calculate_P(s, w)
            
            try:
                res2 = minimize(obj_refine, best_local_params, method='L-BFGS-B', 
                               bounds=bounds, options={'maxiter': 500, 'ftol': 1e-8})
                if res2.fun < best_local_p:
                    s_opt2 = res2.x[:n_pts]
                    w_opt2 = res2.x[n_pts:]
                    update_best(s_opt2, w_opt2)
            except Exception:
                pass

    optimize_small_support(2)
    optimize_small_support(3)
    optimize_small_support(4)
    optimize_small_support(5)
    
    if time.time() < deadline - 60:
        optimize_small_support(6)
    
    if time.time() < deadline - 30:
        optimize_small_support(7)
    
    if time.time() < deadline - 15:
        optimize_small_support(8)
    
    if best_support is None:
        best_support = np.arange(32, dtype=np.float64)
        best_weights = np.ones(32, dtype=np.float64) / 32
        
    return best_support, best_weights
