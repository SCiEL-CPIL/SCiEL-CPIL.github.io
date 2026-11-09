import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def calculate_P(support, weights):
    """
    Calculate P[X1 + X2 + X3 < 2*X4] for discrete distribution.
    Vectorized computation.
    """
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    
    w_sum = np.sum(w)
    if w_sum <= 1e-12:
        return 0.0
    w = w / w_sum
    
    n = len(s)
    if n == 0:
        return 0.0
    
    # S[i,j] = s_i + s_j
    S = s[:, None] + s[None, :]
    # W[i,j] = w_i * w_j
    W = w[:, None] * w[None, :]
    
    # For each (k, l), threshold = 2*s_l - s_k
    # P = sum_{k,l} w_k * w_l * sum_{i,j: S[i,j] < 2*s_l - s_k} W[i,j]
    
    # Vectorize over k: for each k, compute sum over l
    # Threshold matrix: T[k,l] = 2*s_l - s_k
    T = 2.0 * s[None, :] - s[:, None]  # T[k, l] = 2*s_l - s_k
    
    P = 0.0
    for k in range(n):
        # For fixed k, we need sum_l w_l * (sum_{i,j: S[i,j] < T[k,l]} W[i,j])
        # T[k, l] for all l
        thresholds = T[k, :]  # shape (n,)
        
        # For each l, mask = S < thresholds[l]
        # Sum over i,j where S[i,j] < thresholds[l] of W[i,j]
        # This is a 2D comparison: S < thresholds[None, None, :] ... 
        # Actually S is (n,n), thresholds is (n,)
        # We need: for each l, sum_{i,j} W[i,j] * (S[i,j] < thresholds[l])
        
        # Efficient way: sort S and use cumulative sum of W
        # But W is not sorted by S. Let's use the direct approach for small n.
        
        # For n up to ~10, direct is fine
        # mask[l, i, j] = S[i,j] < thresholds[l]
        # This would be n^3 memory. For n=10, that's 1000, fine.
        # For n=20, that's 8000, fine.
        
        # Actually let's do it differently:
        # For each l, compute the inner sum
        inner_sums = np.zeros(n)
        for l in range(n):
            mask = S < thresholds[l]
            inner_sums[l] = np.sum(W * mask)
        
        P += w[k] * np.sum(w * inner_sums)
    
    return P

def _compute_P_fast(support, weights):
    """
    Fast computation of P using sorted pairwise sums.
    """
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    
    w_sum = np.sum(w)
    if w_sum <= 1e-12:
        return 0.0
    w = w / w_sum
    
    n = len(s)
    if n == 0:
        return 0.0
    
    # S[i,j] = s_i + s_j
    S = s[:, None] + s[None, :]
    W = w[:, None] * w[None, :]
    
    # Flatten and sort
    S_flat = S.flatten()
    W_flat = W.flatten()
    
    # Sort by S_flat
    order = np.argsort(S_flat, kind='mergesort')
    S_sorted = S_flat[order]
    W_sorted = W_flat[order]
    W_cumsum = np.cumsum(W_sorted)
    
    # For each threshold t, sum of W where S < t = W_cumsum[bisect_left(S_sorted, t) - 1]
    # Thresholds: T[k,l] = 2*s_l - s_k for all k,l
    T = 2.0 * s[None, :] - s[:, None]  # T[k, l] = 2*s_l - s_k
    T_flat = T.flatten()
    
    # For each threshold, find the index
    # idx[l] = number of elements in S_sorted < T_flat[l]
    # Use searchsorted
    indices = np.searchsorted(S_sorted, T_flat, side='left')
    
    # inner_sum[k,l] = W_cumsum[indices[k,l] - 1] if indices[k,l] > 0 else 0
    # Actually, W_cumsum[i-1] gives sum of first i elements
    # If indices[k,l] = m, then sum of W where S < T is W_cumsum[m-1] if m > 0 else 0
    
    inner_sums = np.zeros_like(T_flat)
    nonzero = indices > 0
    inner_sums[nonzero] = W_cumsum[indices[nonzero] - 1]
    
    inner_sums_2d = inner_sums.reshape(n, n)
    
    # P = sum_{k,l} w_k * w_l * inner_sums_2d[k,l]
    P = np.sum(w[:, None] * w[None, :] * inner_sums_2d)
    
    return P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 600.0  # Safety margin before 630s limit
    
    best_score = -1.0
    best_support = None
    best_weights = None
    
    # Strategy: Use differential evolution to optimize both support and weights
    # for various n values
    
    def objective(params, n):
        """
        params: [t1, ..., t_{n-2}, w0, ..., w_{n-2}]
        Support: [0, t1, ..., t_{n-2}, 1]
        Weights: [w0, ..., w_{n-2}, 1 - sum(w0..w_{n-2})]
        """
        num_internal = n - 2
        t = params[:num_internal]
        w_vars = params[num_internal:]
        
        # Construct support
        s = np.zeros(n)
        s[0] = 0.0
        s[n-1] = 1.0
        if num_internal > 0:
            s[1:n-1] = np.sort(t)
        
        # Construct weights
        w = np.zeros(n)
        w[:n-1] = w_vars
        w[n-1] = 1.0 - np.sum(w_vars)
        
        # Check validity
        if np.any(w < -1e-9) or np.any(s < -1e-9):
            return 1e10
        
        w = np.clip(w, 0, None)
        w_sum = np.sum(w)
        if w_sum <= 1e-12:
            return 1e10
        w = w / w_sum
        
        score = _compute_P_fast(s, w)
        return -score
    
    # Baseline: 2-point optimal
    p_opt = (-1.0 + np.sqrt(17.0)) / 8.0
    s_2pt = np.array([0.0, 1.0])
    w_2pt = np.array([1.0 - p_opt, p_opt])
    score_2pt = _compute_P_fast(s_2pt, w_2pt)
    
    best_score = score_2pt
    best_support = s_2pt.copy()
    best_weights = w_2pt.copy()
    
    # Try n = 3, 4, 5 with differential evolution
    for n in [3, 4, 5]:
        if time.time() > deadline - 60.0:
            break
        
        num_internal = n - 2
        dim = num_internal + (n - 1)  # t variables + w variables
        
        # Bounds: t in [0.01, 0.99], w in [0.01, 0.99]
        bounds = []
        for _ in range(num_internal):
            bounds.append((0.01, 0.99))
        for _ in range(n - 1):
            bounds.append((0.01, 0.99))
        
        # Run differential evolution
        try:
            de_result = differential_evolution(
                objective, bounds, args=(n,),
                maxiter=200, popsize=15,
                tol=1e-8, mutation=(0.5, 1.0),
                recombination=0.7, seed=42,
                init='sobol',
                polish=True,
                callback=None
            )
            
            if de_result.fun < 1e9:
                params_opt = de_result.x
                t_opt = np.sort(params_opt[:num_internal])
                w_vars_opt = np.clip(params_opt[num_internal:], 0, None)
                w_opt = np.zeros(n)
                w_opt[:n-1] = w_vars_opt
                w_opt[n-1] = 1.0 - np.sum(w_vars_opt)
                w_opt = np.clip(w_opt, 0, None)
                w_sum = np.sum(w_opt)
                if w_sum > 1e-12:
                    w_opt = w_opt / w_sum
                    
                    s_opt = np.zeros(n)
                    s_opt[0] = 0.0
                    s_opt[n-1] = 1.0
                    if num_internal > 0:
                        s_opt[1:n-1] = t_opt
                    
                    score = _compute_P_fast(s_opt, w_opt)
                    if score > best_score:
                        best_score = score
                        best_support = s_opt.copy()
                        best_weights = w_opt.copy()
        except Exception:
            pass
        
        if time.time() > deadline - 60.0:
            break
        
        # Also try multiple random restarts with Nelder-Mead for refinement
        num_restarts = 5
        for r in range(num_restarts):
            if time.time() > deadline - 60.0:
                break
            
            # Generate initial guess
            if r == 0:
                t_init = np.linspace(0.1, 0.9, num_internal) if num_internal > 0 else np.array([])
                w_init = np.ones(n-1) / n
            elif r == 1:
                t_init = np.linspace(0.2, 0.8, num_internal) if num_internal > 0 else np.array([])
                w_init = np.zeros(n-1)
                w_init[0] = 0.4
                w_init[n-2] = 0.4
                if n > 3:
                    w_init[1:n-1] = 0.2 / max(1, n-3)
            else:
                t_init = np.sort(np.random.uniform(0.05, 0.95, num_internal)) if num_internal > 0 else np.array([])
                w_init = np.random.uniform(0.1, 1.0, n-1)
                w_init = w_init / np.sum(w_init)
            
            params_init = np.concatenate([t_init, w_init])
            
            try:
                res = minimize(objective, params_init, args=(n,), method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10,
                                      'adaptive': True})
                
                if res.fun < 1e9:
                    t_opt = np.sort(res.x[:num_internal])
                    w_vars_opt = np.clip(res.x[num_internal:], 0, None)
                    w_opt = np.zeros(n)
                    w_opt[:n-1] = w_vars_opt
                    w_opt[n-1] = 1.0 - np.sum(w_vars_opt)
                    w_opt = np.clip(w_opt, 0, None)
                    w_sum = np.sum(w_opt)
                    if w_sum > 1e-12:
                        w_opt = w_opt / w_sum
                        
                        s_opt = np.zeros(n)
                        s_opt[0] = 0.0
                        s_opt[n-1] = 1.0
                        if num_internal > 0:
                            s_opt[1:n-1] = t_opt
                        
                        score = _compute_P_fast(s_opt, w_opt)
                        if score > best_score:
                            best_score = score
                            best_support = s_opt.copy()
                            best_weights = w_opt.copy()
            except Exception:
                pass
    
    # Ensure validity
    best_support = np.asarray(best_support, dtype=np.float64)
    best_weights = np.asarray(best_weights, dtype=np.float64)
    
    # Flatten
    best_support = best_support.flatten()
    best_weights = best_weights.flatten()
    
    # Final sanity checks
    if len(best_support) != len(best_weights) or len(best_support) < 1:
        # Fallback to 2-point
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([1.0 - p_opt, p_opt])
        
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    
    # Ensure weights sum to positive
    w_sum = np.sum(best_weights)
    if w_sum <= 1e-12:
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])
    
    return best_support, best_weights
