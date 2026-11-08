import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_genlaguerre, roots_genlaguerre
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 630s kill

    m = 12
    alpha = 1

    # Precompute the transformation matrix from monomial basis to Laguerre basis
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    A = np.zeros((m + 1, m + 1))
    for k in range(m + 1):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    # Solve for c given p (monomial coefficients, ascending). c = (A^T)^{-1} p
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

    def compute_norm_sq(roots):
        """
        Compute the squared L2 norm with weight e^{-x} of P(x) = prod(x - r_i).
        """
        # Build polynomial coefficients in standard basis (ascending)
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1]  # ascending
        
        # c = A_T_inv @ p
        c = A_T_inv @ p
        
        # Norm squared: ||P||^2 = sum c_k^2 * (k+1) for alpha=1
        norm_sq = np.sum(c**2 * (np.arange(m + 1) + 1))
        
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def cost_func(log_roots):
        """
        Minimize the upper bound on C4.
        Bound = ||P||^2 / (product of roots)
        """
        roots = np.exp(log_roots)
        
        # Check for numerical issues
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
            
        norm_sq = compute_norm_sq(roots)
        prod_roots = np.prod(roots)
        
        if norm_sq <= 1e-10 or prod_roots <= 1e-10:
            return 1e10
            
        return norm_sq / prod_roots

    # Generate diverse seed configurations
    seeds_configs = []
    
    # 1. Geometric sequence
    seeds_configs.append(4.0 * (2.0 ** np.arange(12)))
    
    # 2. Zeros of L_12^(1) - natural starting points
    try:
        laguerre_roots = roots_genlaguerre(m, alpha)[0]
        seeds_configs.append(laguerre_roots.copy())
        for scale in [0.5, 1.5, 2.0, 3.0]:
            seeds_configs.append(laguerre_roots * scale)
    except Exception:
        pass
    
    # 3. Linear sequence
    seeds_configs.append(np.linspace(1, 20, 12))
    
    # 4. Exponential sequence
    seeds_configs.append(np.exp(np.linspace(0, 3, 12)))
    
    # 5. Mixed: some close, some far
    mixed = np.concatenate([np.linspace(0.5, 2, 6), np.linspace(5, 30, 6)])
    seeds_configs.append(mixed)
    
    # 6. Clustered
    clustered = np.concatenate([1 + np.linspace(0, 0.5, 6), 5 + np.linspace(0, 0.5, 6)])
    seeds_configs.append(clustered)
    
    best_roots = None
    best_val = 1e10
    best_log_roots = None

    # Evaluate all seeds
    for seed_roots in seeds_configs:
        if np.any(seed_roots <= 0) or np.any(np.isnan(seed_roots)):
            continue
        log_roots = np.log(seed_roots)
        val = cost_func(log_roots)
        if val < best_val:
            best_val = val
            best_roots = seed_roots.copy()
            best_log_roots = log_roots.copy()

    # Determine bounds from seed performance
    all_log_roots = [np.log(sr) for sr in seeds_configs if np.all(sr > 0) and not np.any(np.isnan(sr))]
    if all_log_roots:
        min_log = max(-4.0, min(np.min(lr) for lr in all_log_roots) - 2.0)
        max_log = min(10.0, max(np.max(lr) for lr in all_log_roots) + 2.0)
    else:
        min_log, max_log = -3.0, 8.0
    
    bounds = [(min_log, max_log)] * 12

    # Run Differential Evolution with more restarts, fewer iterations each
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808]
    
    for seed in seeds:
        elapsed = time.time() - start_time
        if elapsed > deadline - 60:  # Leave 60s for polish
            break
        
        try:
            result = differential_evolution(
                cost_func, 
                bounds, 
                seed=seed, 
                maxiter=80, 
                popsize=12, 
                tol=1e-12, 
                polish=True,
                init='sobol'
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x.copy()
                best_roots = np.exp(best_log_roots)
                
        except Exception as e:
            pass

    # Final high-precision polish pass
    if time.time() - start_time < deadline - 10 and best_log_roots is not None:
        # L-BFGS-B polish
        try:
            polish_result = minimize(
                cost_func, best_log_roots, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-16}
            )
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_log_roots = polish_result.x.copy()
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass
        
        # Nelder-Mead polish from best point
        if time.time() - start_time < deadline - 5:
            try:
                nm_result = minimize(
                    cost_func, best_log_roots, method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
                )
                if nm_result.fun < best_val:
                    best_val = nm_result.fun
                    best_log_roots = nm_result.x.copy()
                    best_roots = np.exp(best_log_roots)
            except Exception:
                pass

    # Ensure roots are positive
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)
    
    # Absolute fallback
    return [4.0 * (2.0 ** i) for i in range(12)]
