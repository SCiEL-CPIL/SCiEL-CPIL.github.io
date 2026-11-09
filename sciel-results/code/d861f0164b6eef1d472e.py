import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_genlaguerre, roots_laguerre
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
        p = coeffs_desc[::-1]  # ascending
        
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

    # Primary candidate: roots of the Laguerre polynomial L_12(x)
    # This is the mathematically natural choice for the Cohn-Gonçalves construction
    laguerre_roots = roots_laguerre(12)[0]
    laguerre_roots = np.sort(laguerre_roots)
    laguerre_log_roots = np.log(laguerre_roots)
    
    # Evaluate the Laguerre roots candidate
    laguerre_val = cost_func(laguerre_log_roots)
    
    best_roots = laguerre_roots.copy()
    best_val = laguerre_val
    best_log_roots = laguerre_log_roots.copy()

    # Refinement: try to improve via local optimization around the Laguerre roots
    # Use L-BFGS-B with tight tolerances for high-precision polish
    if time.time() - start_time < deadline - 5:
        try:
            # Define bounds centered around the Laguerre roots with reasonable spread
            # The Laguerre roots range from ~0.5 to ~40, so log range is ~[-0.7, 3.7]
            # Allow some flexibility: log(0.1) to log(100)
            bounds = [(-2.0, 5.0)] * 12
            
            polish_result = minimize(
                cost_func, best_log_roots, method='L-BFGS-B',
                bounds=bounds, 
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-16, 'maxls': 50}
            )
            
            if polish_result.fun < best_val and not np.any(np.isnan(polish_result.x)):
                best_val = polish_result.fun
                best_log_roots = polish_result.x
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass

    # Secondary: if local polish didn't improve much, try a broader search
    # but only if we have time remaining
    elapsed = time.time() - start_time
    if elapsed < deadline - 30 and best_val > 0.5 * laguerre_val:
        # The Laguerre roots are a strong candidate; do a limited global search
        # to verify no significantly better solution exists
        try:
            # Use a smaller, faster DE search
            result = differential_evolution(
                cost_func, 
                [(-2.0, 5.0)] * 12, 
                seed=42, 
                maxiter=50, 
                popsize=10, 
                tol=1e-8, 
                polish=False,
                init='sobol'
            )
            
            if result.fun < best_val and not np.any(np.isnan(result.x)):
                best_val = result.fun
                best_log_roots = result.x
                best_roots = np.exp(best_log_roots)
                
                # Polish the DE result
                if time.time() - start_time < deadline - 5:
                    polish2 = minimize(
                        cost_func, best_log_roots, method='L-BFGS-B',
                        bounds=[(-2.0, 5.0)] * 12,
                        options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15}
                    )
                    if polish2.fun < best_val and not np.any(np.isnan(polish2.x)):
                        best_val = polish2.fun
                        best_log_roots = polish2.x
                        best_roots = np.exp(best_log_roots)
        except Exception:
            pass

    # Ensure roots are positive
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)
    
    # Absolute fallback: Laguerre roots
    return list(np.sort(roots_laguerre(12)[0]))
