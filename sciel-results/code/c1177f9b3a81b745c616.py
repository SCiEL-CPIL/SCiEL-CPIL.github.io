import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill

    m = 12
    alpha = 1

    # Precompute the transformation matrix from monomial basis to Laguerre basis
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    A = np.zeros((m + 1, m + 1))
    for k in range(m + 1):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k + 1, j + 1) / math.factorial(j)

    # Norm vector d_k = (k+1)
    d = np.arange(m + 1) + 1

    # We want to compute c from p. p = A^T c => c = (A^T)^-1 p
    # Then norm^2 = c^T diag(d) c = p^T (A^T)^-T diag(d) (A^T)^-1 p
    # Let K = (A^T)^-T diag(d) (A^T)^-1 = A^-1 diag(d) (A^T)^-1
    try:
        A_inv = np.linalg.inv(A)
        K = A_inv @ np.diag(d) @ A_inv.T
    except np.linalg.LinAlgError:
        A_pinv = np.linalg.pinv(A)
        K = A_pinv @ np.diag(d) @ A_pinv.T

    # Precompute factorials for faster polynomial coefficient computation
    # Instead of np.poly, we'll use a more stable approach
    # For P(x) = prod_{i=1}^{12} (x - r_i), we can compute coefficients iteratively
    
    def compute_norm_sq_fast(roots):
        """
        Compute the squared L2 norm with weight e^{-x} of P(x) = prod(x - r_i).
        Uses iterative polynomial multiplication for stability.
        """
        # Build polynomial coefficients in ascending order: [c0, c1, ..., c12]
        # Start with p = [1] (constant 1)
        p = np.array([1.0])
        for r in roots:
            # Multiply p by (x - r)
            new_p = np.zeros(len(p) + 1)
            new_p[:-1] = -r * p
            new_p[1:] += p
            p = new_p
        
        # p should now have length m+1 = 13
        if len(p) != m + 1:
            return 0.0
            
        # Compute norm squared = p @ K @ p
        norm_sq = float(p @ K @ p)

        if norm_sq < 0:  # Numerical error check
            return 0.0
        return norm_sq

    def cost_func(log_roots):
        """
        Minimize the upper bound on C4.
        The objective is 1/norm_sq (minimize this = maximize norm_sq).
        """
        roots = np.exp(log_roots)
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10

        norm_sq = compute_norm_sq_fast(roots)
        if norm_sq <= 1e-10:
            return 1e10
        return 1.0 / norm_sq

    # Optimization Strategy:
    # 1. Use the precomputed K matrix for faster norm calculation.
    # 2. Use diverse local optimizations from multiple seeds.
    # 3. Use a single DE run with polish, then final high-precision polish.
    
    # Seed: Geometric sequence a=4, q=2 (from Current Program)
    seed_log_roots = np.log(4.0) + np.arange(12) * np.log(2.0)
    
    # Bounds for log_roots
    bounds = [(-4.0, 10.0)] * 12

    best_roots = None
    best_val = 1e10
    best_log_roots = None

    # First, evaluate the seed to establish a baseline
    seed_val = cost_func(seed_log_roots)
    best_val = seed_val
    best_roots = np.exp(seed_log_roots)
    best_log_roots = seed_log_roots.copy()

    # Try local optimizations from the seed and some other reasonable starts
    # This is cheap and might find a better local minimum near the seed.
    
    # Additional initial guesses
    # Geometric in [1, 10]
    r1 = 1.0 * (10.0 ** (np.arange(12) / 11.0))
    seeds = [seed_log_roots, np.log(r1)]
    
    # Roots of Laguerre Polynomial L_12^(1)
    # L_n^(alpha)(x) = \sum_{k=0}^n (-1)^k \binom{n+alpha}{n-k} \frac{x^k}{k!}
    lag_coeffs_desc = np.zeros(13)
    for k in range(13):
        if k <= 12:
            lag_coeffs_desc[12 - k] = ((-1)**k) * math.comb(13, 12 - k) / math.factorial(k)
            
    try:
        lag_roots = np.roots(lag_coeffs_desc)
        lag_roots_real = lag_roots[np.abs(lag_roots.imag) < 1e-9].real
        lag_roots_pos = lag_roots_real[lag_roots_real > 0]
        if len(lag_roots_pos) == 12:
            seeds.append(np.log(lag_roots_pos))
    except Exception:
        pass
    
    # Also try roots scaled by different factors
    if len(seeds) > 2:  # If we have Laguerre roots
        lag_seed = seeds[2]
        for scale in [0.5, 1.5, 2.0, 3.0]:
            seeds.append(lag_seed + np.log(scale))

    # Local polish from each seed
    for seed_log in seeds:
        if time.time() > deadline - 120:
            break
        try:
            res = minimize(
                cost_func, seed_log, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_roots = np.exp(res.x)
                best_log_roots = res.x.copy()
        except Exception:
            pass

    # Global Search with Differential Evolution - single run with more iterations
    if time.time() < deadline - 60:
        try:
            result = differential_evolution(
                cost_func, 
                bounds, 
                seed=123, 
                maxiter=300, 
                popsize=15, 
                tol=1e-12, 
                polish=True, 
                init='sobol',
                update_interval=10,
                callback=None
            )
            if result.fun < best_val:
                best_val = result.fun
                best_roots = np.exp(result.x)
                best_log_roots = result.x.copy()
        except Exception:
            pass

    # Final high-precision polish pass
    if best_log_roots is not None and time.time() < deadline - 10:
        try:
            polish_result = minimize(
                cost_func, best_log_roots, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-16}
            )
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_roots = np.exp(polish_result.x)
                best_log_roots = polish_result.x.copy()
        except Exception:
            pass

    # Ensure roots are positive and sorted (though order doesn't matter for the product)
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)
    
    # Absolute fallback
    return [4.0 * (2.0 ** i) for i in range(12)]
