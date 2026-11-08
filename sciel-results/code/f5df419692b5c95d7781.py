import numpy as np
from scipy.optimize import differential_evolution, minimize, basin_hopping
from scipy.special import eval_genlaguerre
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin before 600s limit

    m = 12
    n = 1

    # Precompute the transformation matrix from standard basis to Laguerre basis
    # P(x) = sum c_k L_k^{(1)}(x)
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    
    A = np.zeros((13, 13))  # A[k, j] = coeff of x^j in L_k^{(1)}(x)
    for k in range(13):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    # Solve for c given p. c = (A^T)^{-1} p
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

    def compute_norm_sq(roots):
        """
        Compute the squared norm of the polynomial in the Laguerre basis.
        roots: array of 12 positive numbers.
        """
        # Build polynomial coefficients in standard basis (ascending)
        # P(x) = prod(x - r_i)
        # np.poly returns descending coeffs: [1, c1, ..., cm]
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1] # ascending
        
        # c = A_T_inv @ p
        c = A_T_inv @ p
        
        # Norm squared: <L_k, L_l> = delta_kl * (k+1) for alpha=1
        # ||P||^2 = sum c_k^2 * (k+1)
        norm_sq = np.sum(c**2 * (np.arange(13) + 1))
        
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def cost_func(log_roots):
        """
        Minimize the upper bound on C4.
        Bound ~ 1 / ||P||^2.
        So we minimize 1/norm_sq, which is equivalent to maximizing norm_sq.
        We return 1/norm_sq.
        """
        roots = np.exp(log_roots)
        
        # Check for numerical issues
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
            
        norm_sq = compute_norm_sq(roots)
        
        if norm_sq <= 1e-10:
            return 1e10
            
        return 1.0 / norm_sq

    # Seed: geometric sequence a=4, q=2
    # r_i = 4 * 2^i
    # log_r_i = log(4) + i * log(2)
    seed_log_roots = np.log(4.0) + np.arange(12) * np.log(2.0)
    
    # Bounds for log_roots
    bounds = [(-4.0, 10.0)] * 12

    best_roots = None
    best_val = 1e10
    best_log_roots = None

    # Strategy: Multi-restart L-BFGS-B from diverse initial points
    # This is much faster than DE and allows many more restarts
    
    # Generate diverse starting points
    rng = np.random.RandomState(42)
    start_points = [seed_log_roots.copy()]
    
    # Add perturbations of the seed
    for scale in [0.5, 0.7, 1.0, 1.3, 1.5, 2.0]:
        pt = seed_log_roots * scale + rng.uniform(-0.5, 0.5, 12)
        start_points.append(np.clip(pt, -4.0, 10.0))
    
    # Add random points
    for _ in range(20):
        pt = rng.uniform(-3.0, 9.0, 12)
        start_points.append(pt)
    
    # Add points based on different geometric ratios
    for q in [1.5, 1.8, 2.2, 2.5, 3.0]:
        for a in [2.0, 4.0, 8.0]:
            pt = np.log(a) + np.arange(12) * np.log(q)
            pt = np.clip(pt, -4.0, 10.0)
            start_points.append(pt)

    # Evaluate seed first
    seed_val = cost_func(seed_log_roots)
    best_val = seed_val
    best_roots = np.exp(seed_log_roots)
    best_log_roots = seed_log_roots.copy()

    # Run L-BFGS-B from each starting point
    for i, x0 in enumerate(start_points):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
        
        try:
            result = minimize(
                cost_func, x0, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x.copy()
                best_roots = np.exp(best_log_roots)
                
        except Exception:
            pass

    # If we still have time, try a few DE runs with small population for global exploration
    if time.time() - start_time < deadline - 30:
        for seed in [42, 123, 456]:
            elapsed = time.time() - start_time
            if elapsed > deadline - 15:
                break
            try:
                result = differential_evolution(
                    cost_func, 
                    bounds, 
                    seed=seed, 
                    maxiter=100, 
                    popsize=10, 
                    tol=1e-10, 
                    polish=False,
                    init='sobol'
                )
                
                if result.fun < best_val:
                    best_val = result.fun
                    best_log_roots = result.x.copy()
                    best_roots = np.exp(best_log_roots)
                    
                # Polish the DE result
                if time.time() - start_time < deadline - 5:
                    polish_result = minimize(
                        cost_func, result.x, method='L-BFGS-B',
                        bounds=bounds, options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-14}
                    )
                    if polish_result.fun < best_val:
                        best_val = polish_result.fun
                        best_log_roots = polish_result.x.copy()
                        best_roots = np.exp(best_log_roots)
                        
            except Exception:
                pass

    # Final high-precision polish pass on the best result
    if time.time() - start_time < deadline - 2 and best_log_roots is not None:
        try:
            polish_result = minimize(
                cost_func, best_log_roots, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_log_roots = polish_result.x
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass

    # Ensure roots are positive and sorted (though order doesn't matter for the product)
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)
    
    # Absolute fallback
    return [4.0 * (2.0 ** i) for i in range(12)]
