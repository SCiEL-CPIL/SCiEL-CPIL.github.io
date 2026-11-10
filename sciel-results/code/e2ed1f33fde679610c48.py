import numpy as np
import time

def compute_P(s, w):
    """
    Computes P[X1 + X2 + X3 < 2*X4] efficiently.
    s: support points (n,)
    w: weights (n,)
    """
    n = len(s)
    if n == 0:
        return 0.0
        
    # Pairwise sums S_ij = s_i + s_j
    S_ij = s[:, np.newaxis] + s[np.newaxis, :]
    # Pairwise weight products W_ij = w_i * w_j
    W_ij = w[:, np.newaxis] * w[np.newaxis, :]
    
    # Flatten and sort
    S_flat = S_ij.flatten()
    W_flat = W_ij.flatten()
    
    # Sort by S_ij
    sort_idx = np.argsort(S_flat)
    S_sorted = S_flat[sort_idx]
    W_sorted = W_flat[sort_idx]
    
    # Cumulative sum of weights
    W_cumsum = np.cumsum(W_sorted)
    
    P = 0.0
    for l in range(n):
        # Thresholds T_k = 2*s_l - s_k for all k
        T_k = 2 * s[l] - s
        # Find number of pairs (i,j) such that S_ij < T_k
        # np.searchsorted returns index of first element >= T_k
        # So index is the count of elements < T_k
        counts = np.searchsorted(S_sorted, T_k, side='left')
        # The sum of W_ij for S_ij < T_k is W_cumsum[counts - 1]
        # Need to handle counts == 0
        Q_l = np.zeros(n)
        mask = counts > 0
        Q_l[mask] = W_cumsum[counts[mask] - 1]
        
        # P_l = sum_k w_k * Q_l[k]
        P_l = np.dot(w, Q_l)
        P += w[l] * P_l
        
    return P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 580.0
    
    best_s = None
    best_w = None
    best_P = -1.0
    
    def update_best(s, w):
        nonlocal best_P, best_s, best_w
        if s is None or w is None:
            return
        try:
            p_val = compute_P(s, w)
            if p_val > best_P:
                best_P = p_val
                best_s = s.copy()
                best_w = w.copy()
        except Exception:
            pass

    # 1. Try the known good uniform grid with weight optimization (fast, high score)
    # The current program's score of 1.3252 likely comes from this or a similar small-n optimization.
    # We will use a faster compute_P to allow more iterations or larger n.
    
    # Start with a few promising configurations
    # Configuration 1: Uniform grid n=32
    s_init = np.arange(32, dtype=np.float64)
    w_init = np.ones(32) / 32
    update_best(s_init, w_init)
    
    # Configuration 2: Small support with optimized weights
    # Try n=16, n=32 with SLSQP weight optimization
    for n_grid in [16, 32]:
        if time.time() > deadline - 10:
            break
            
        s = np.arange(n_grid, dtype=np.float64)
        w = np.ones(n_grid) / n_grid
        
        # Simple local search on weights
        # Since compute_P is fast now, we can do a few gradient steps or just keep the uniform if it's good.
        # Actually, let's try to optimize weights for a fixed support using a simple method.
        # But the main gain might be from optimizing the support itself for small n.
        
    # 2. Structural change: Optimize support AND weights for small n using Nelder-Mead
    # This explores the "shape" of the distribution.
    # The inspirations suggest that small n (2-5) with optimized positions can yield high scores.
    
    def optimize_small_support(n_pts):
        if time.time() > deadline - 5:
            return
        
        # Initial guess: log-spaced or linear
        if n_pts == 2:
            s_init = np.array([0.0, 10.0])
            w_init = np.array([0.5, 0.5])
        elif n_pts == 3:
            s_init = np.array([0.0, 1.0, 10.0])
            w_init = np.array([1/3, 1/3, 1/3])
        elif n_pts == 4:
            s_init = np.array([0.0, 0.5, 2.0, 10.0])
            w_init = np.ones(4)/4
        elif n_pts == 5:
            s_init = np.array([0.0, 0.5, 2.0, 5.0, 10.0])
            w_init = np.ones(5)/5
        else:
            return
        
        def obj_small(params):
            s = params[:n_pts]
            w = params[n_pts:]
            
            # Enforce non-negative supports
            s = np.clip(s, 0, 1e6)
            w = np.clip(w, 0, 1)
            w_sum = np.sum(w)
            if w_sum < 1e-9:
                return 0.0
            w = w / w_sum
            
            return -compute_P(s, w)
        
        bounds_s = [(0, 1000)] * n_pts
        bounds_w = [(0, 1)] * n_pts
        bounds = bounds_s + bounds_w
        
        # Start from a few random-ish initializations
        best_local_p = -1.0
        best_local_params = None
        
        rng = np.random.default_rng(42)
        
        # Use Nelder-Mead for local optimization
        # It's derivative-free and robust for small dimensions
        for restart in range(5):
            if time.time() > deadline - 5:
                break
                
            # Perturb initial guess
            s_pert = s_init * (1 + 0.1 * rng.standard_normal(n_pts))
            s_pert = np.clip(s_pert, 0, 1000)
            w_pert = w_init * (1 + 0.1 * rng.standard_normal(n_pts))
            w_pert = np.clip(w_pert, 0.01, 1)
            
            params_init = np.concatenate([s_pert, w_pert])
            
            try:
                from scipy.optimize import minimize
                res = minimize(obj_small, params_init, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
                if res.fun < best_local_p:
                    best_local_p = res.fun
                    best_local_params = res.x
            except Exception:
                pass
            
        if best_local_params is not None:
            s_opt = best_local_params[:n_pts]
            w_opt = best_local_params[n_pts:]
            update_best(s_opt, w_opt)

    # Run optimization for small support sizes
    for n in [2, 3, 4, 5]:
        optimize_small_support(n)
        
    # 3. If we have time, try a slightly larger n with SLSQP alternating optimization
    # This combines the "grid" structure with local refinement.
    if time.time() < deadline - 30:
        n = 16
        s = np.linspace(0, 10, n)
        w = np.ones(n) / n
        
        # Alternate optimizing s and w
        for _ in range(5):
            if time.time() > deadline - 10:
                break
                
            # Optimize w
            try:
                from scipy.optimize import minimize
                def obj_w(params):
                    return -compute_P(s, params)
                res_w = minimize(obj_w, w, method='SLSQP', bounds=[(0, 1)]*n, 
                                  constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}],
                                  options={'maxiter': 50, 'ftol': 1e-6})
                if res_w.success:
                    w_new = np.maximum(res_w.x, 0)
                    w_new = w_new / np.sum(w_new)
                    w = w_new
            except:
                pass
            
            # Optimize s
            try:
                def obj_s(params):
                    return -compute_P(params, w)
                res_s = minimize(obj_s, s, method='SLSQP', bounds=[(0, 100)]*n,
                                  options={'maxiter': 50, 'ftol': 1e-6})
                if res_s.success:
                    s_new = np.maximum(res_s.x, 0)
                    s = s_new
            except:
                pass
            
            update_best(s, w)

    if best_s is None:
        best_s = np.arange(32, dtype=np.float64)
        best_w = np.ones(32, dtype=np.float64) / 32
        
    return best_s, best_w
