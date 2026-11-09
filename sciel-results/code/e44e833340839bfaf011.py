import numpy as np
import time
from scipy.optimize import linprog

def calculate_P(support, weights):
    """
    Calculate P[X1 + X2 + X3 < 2*X4] for discrete distribution.
    """
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    
    # Normalize weights
    w_sum = np.sum(w)
    if w_sum <= 1e-12:
        return 0.0
    w = w / w_sum
    
    n = len(s)
    if n == 0:
        return 0.0
        
    # Precompute pairwise sums and weights
    # s_i + s_j
    S = s[:, None] + s[None, :]
    W = w[:, None] * w[None, :]
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * s[l] - s[k]
            # Sum W_ij where S_ij < threshold
            mask = S < threshold
            P += w[k] * w[l] * np.sum(W[mask])
            
    return P

def _compute_P_fast(support, weights):
    """
    Fast computation of P using vectorized operations.
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
    
    # For each (k, l), we need sum_{i,j: S[i,j] < 2*s_l - s_k} w_i * w_j
    # This is a 4D problem: P = sum_{k,l} w_k * w_l * (sum_{i,j: S[i,j] < 2*s_l - s_k} w_i * w_j)
    
    # Let's compute it more efficiently
    # P = sum_k w_k * sum_l w_l * F(2*s_l - s_k)
    # where F(t) = sum_{i,j: S[i,j] < t} w_i * w_j
    
    # Precompute F for all thresholds
    # Thresholds are 2*s_l - s_k for all k, l
    # There are n^2 thresholds, but many may be the same
    
    # Actually, let's just do the double sum but with precomputed sorted structure
    # For now, use the same approach as calculate_P but ensure it's fast
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * s[l] - s[k]
            mask = S < threshold
            P += w[k] * w[l] * np.sum(w[:, None] * w[None, :][mask])
            
    return P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin
    
    best_score = -1.0
    best_support = None
    best_weights = None
    
    # Strategy: Use linear programming for fixed support grids
    # For a fixed support s = [s_0, ..., s_{n-1}], the problem of maximizing
    # P[X1+X2+X3 < 2*X4] over weights w is NOT linear because P involves
    # products of weights (w_k * w_l * w_i * w_j).
    # 
    # However, we can use a different approach:
    # 1. Try various fixed support structures
    # 2. For each, use optimization to find good weights
    # 3. Also try to optimize the support points themselves
    #
    # Actually, let's think about this differently. The objective is a quartic
    # polynomial in the weights. This is a non-convex optimization problem.
    # 
    # Alternative approach: Use the fact that for 2-point distributions {0, 1},
    # we can derive the optimal analytically. For more points, we need numerical
    # optimization.
    
    # First, get the 2-point optimal as a baseline
    # For {0, 1} with weight p on 1:
    # P = p * [(1-p)^3 + 3*p*(1-p)^2] = p*(1-p)^2*(1+2p)
    # Maximize f(p) = p*(1-p)^2*(1+2p)
    # f'(p) = (1-p)(1-p-4p^2) = 0
    # p = 1 or 4p^2 + p - 1 = 0
    # p = (-1 + sqrt(1+16))/8 = (-1+sqrt(17))/8 ≈ 0.3904
    
    p_opt = (-1.0 + np.sqrt(17.0)) / 8.0
    s_2pt = np.array([0.0, 1.0])
    w_2pt = np.array([1.0 - p_opt, p_opt])
    score_2pt = calculate_P(s_2pt, w_2pt)
    
    best_score = score_2pt
    best_support = s_2pt.copy()
    best_weights = w_2pt.copy()
    
    # Now try to improve with more support points
    # Use a structured approach: try supports like {0, a, 1} or {0, a, b, 1}
    
    for n in [3, 4, 5, 6]:
        if time.time() > deadline:
            break
            
        # Try different support structures
        # Structure 1: {0, t, 1} for n=3
        # Structure 2: {0, t1, t2, 1} for n=4
        # etc.
        
        # For each structure, optimize both support and weights
        # This is a non-convex problem, so we'll use multiple restarts
        
        num_internal = n - 2  # internal support points (excluding 0 and 1)
        
        # Define the full optimization: variables are [t1, ..., t_{n-2}, w0, ..., w_{n-2}]
        # w_{n-1} = 1 - sum(w0, ..., w_{n-2})
        # s = [0, t1, ..., t_{n-2}, 1]
        
        def objective(params):
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
            
            return -calculate_P(s, w)
        
        # Multiple restarts
        num_restarts = 8
        for r in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Generate initial guess
            if r == 0:
                # Uniform
                t_init = np.linspace(0.1, 0.9, num_internal) if num_internal > 0 else np.array([])
                w_init = np.ones(n-1) / n
            elif r == 1:
                # Concentrate on endpoints
                t_init = np.linspace(0.2, 0.8, num_internal) if num_internal > 0 else np.array([])
                w_init = np.zeros(n-1)
                w_init[0] = 0.4
                w_init[n-2] = 0.4
                if n > 3:
                    w_init[1:n-1] = 0.2 / max(1, n-3)
            else:
                # Random
                t_init = np.sort(np.random.uniform(0.05, 0.95, num_internal)) if num_internal > 0 else np.array([])
                w_init = np.random.uniform(0.1, 1.0, n-1)
                w_init = w_init / np.sum(w_init)
            
            params_init = np.concatenate([t_init, w_init])
            
            try:
                from scipy.optimize import minimize
                res = minimize(objective, params_init, method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9,
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
                        
                        score = calculate_P(s_opt, w_opt)
                        if score > best_score:
                            best_score = score
                            best_support = s_opt.copy()
                            best_weights = w_opt.copy()
            except Exception:
                pass
    
    # Also try some structured weight distributions on fixed supports
    # For support {0, 1/n, 2/n, ..., 1}, try different weight patterns
    
    for n in [4, 6, 8]:
        if time.time() > deadline:
            break
            
        s_fixed = np.linspace(0.0, 1.0, n)
        
        # Try exponential weights
        for alpha in [0.5, 1.0, 2.0, 3.0]:
            w_exp = np.exp(-alpha * s_fixed)
            w_exp = w_exp / np.sum(w_exp)
            score = calculate_P(s_fixed, w_exp)
            if score > best_score:
                best_score = score
                best_support = s_fixed.copy()
                best_weights = w_exp.copy()
        
        # Try power-law weights
        for beta in [0.5, 1.0, 2.0]:
            w_pow = s_fixed ** beta
            w_pow = w_pow / np.sum(w_pow)
            score = calculate_P(s_fixed, w_pow)
            if score > best_score:
                best_score = score
                best_support = s_fixed.copy()
                best_weights = w_pow.copy()
    
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
