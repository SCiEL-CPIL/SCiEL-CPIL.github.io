import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_genlaguerre
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin before 600s limit

    m = 12
    n = 1
    alpha = 1

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
        
    # Precompute the kernel matrix K such that norm_sq = p^T K p
    # norm_sq = c^T diag(k+1) c
    # c = A_T_inv p
    # norm_sq = p^T A_T_inv^T diag(k+1) A_T_inv p
    # Let K = A_T_inv^T diag(k+1) A_T_inv
    # Note: A is upper triangular. A_T is lower triangular.
    # We can just compute K directly for speed in the loop.
    d = np.arange(13) + 1
    K = A_T_inv.T @ np.diag(d) @ A_T_inv

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
        
        # norm_sq = p @ K @ p
        # Using float() to ensure scalar return for optimization
        norm_sq = float(p @ K @ p)
        
        if norm_sq <= 1e-10 or np.isnan(norm_sq):
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

    # Optimization Strategy:
    # 1. Evaluate seed (geometric sequence)
    # 2. Local polish from seed and other good starts (Laguerre roots)
    # 3. Global search with Differential Evolution (single long run or few restarts)
    # 4. Final high-precision polish

    # Seed: geometric sequence a=4, q=2
    # r_i = 4 * 2^i
    # log_r_i = log(4) + i * log(2)
    seed_log_roots = np.log(4.0) + np.arange(12) * np.log(2.0)
    
    # Bounds for log_roots
    # Roots should be positive.
    # Based on seed (4 to 4*2^11 ~ 8192), log range is [~1.4, ~9].
    # Let's allow a wider range: log(0.01) ~ -4.6 to log(10000) ~ 9.2
    bounds = [(-4.0, 10.0)] * 12
    
    best_roots = None
    best_val = 1e10
    best_log_roots = None

    # 1. Evaluate Seed
    seed_val = cost_func(seed_log_roots)
    best_val = seed_val
    best_roots = np.exp(seed_log_roots)
    best_log_roots = seed_log_roots.copy()

    # 2. Additional Initial Guesses
    # Roots of Laguerre Polynomial L_12^(1)
    # These are the zeros of the Laguerre polynomial, which might be a good starting point
    # or a local minimum.
    lag_coeffs_desc = np.zeros(13)
    for k in range(13):
        if k <= 12:
            lag_coeffs_desc[12 - k] = ((-1)**k) * math.comb(13, 12 - k) / math.factorial(k)
            
    try:
        lag_roots = np.roots(lag_coeffs_desc)
        lag_roots_real = lag_roots[np.abs(lag_roots.imag) < 1e-9].real
        lag_roots_pos = lag_roots_real[lag_roots_real > 0]
        if len(lag_roots_pos) == 12:
            lag_log_roots = np.log(lag_roots_pos)
            lag_val = cost_func(lag_log_roots)
            if lag_val < best_val:
                best_val = lag_val
                best_roots = lag_roots_pos.copy()
                best_log_roots = lag_log_roots.copy()
    except Exception:
        pass

    # 3. Local Polish from Best Candidates
    # We polish the seed and the Laguerre roots separately to see if we can improve.
    candidates = [seed_log_roots]
    if 'lag_log_roots' in locals():
        candidates.append(lag_log_roots)
        
    # Add a few random points in log space within bounds to diversify local search
    np.random.seed(42)
    for _ in range(5):
        rand_log_roots = np.random.uniform(bounds[0][0], bounds[0][1], 12)
        candidates.append(rand_log_roots)

    for cand in candidates:
        if time.time() > deadline - 30:
            break
        try:
            res = minimize(
                cost_func, cand, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_roots = np.exp(res.x)
                best_log_roots = res.x
        except Exception:
            pass

    # 4. Global Search with Differential Evolution
    # Use a single DE run with sufficient population and iterations.
    # If time permits, do a second run with a different seed.
    
    de_configs = [
        {'seed': 123, 'maxiter': 300, 'popsize': 25},
        {'seed': 456, 'maxiter': 200, 'popsize': 20}
    ]

    for cfg in de_configs:
        if time.time() > deadline - 60:
            break
        try:
            result = differential_evolution(
                cost_func, 
                bounds, 
                seed=cfg['seed'], 
                maxiter=cfg['maxiter'], 
                popsize=cfg['popsize'], 
                tol=1e-10, 
                polish=True,  # L-BFGS-B polish
                init='sobol' # Better initial distribution
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x
                best_roots = np.exp(best_log_roots)
                
        except Exception:
            pass

    # 5. Final high-precision polish pass
    if time.time() - start_time < deadline - 10 and best_log_roots is not None:
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
