import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_genlaguerre
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill

    m = 12

    # Precompute the transformation matrix from monomial basis to Laguerre basis
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    
    A = np.zeros((m + 1, m + 1))  # A[k, j] = coeff of x^j in L_k^{(1)}(x)
    for k in range(m + 1):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    # Precompute inverse for converting back from Laguerre to monomial basis
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        A_inv = np.linalg.pinv(A)
    
    # Precompute the weight vector for norm calculation
    weight_vec = np.arange(m + 1) + 1  # (k+1) for k=0..m

    def laguerre_to_roots(c):
        """Convert Laguerre coefficients to monomial coefficients and find roots."""
        # c is in Laguerre basis: P(x) = sum c_k L_k^(1)(x)
        # Convert to monomial basis: p = A @ c
        p = A @ c  # monomial coefficients (ascending)
        # Find roots
        try:
            roots = np.roots(p[::-1])  # np.roots expects descending
        except Exception:
            return None, None
        return p, roots

    def compute_norm_sq_lag(c):
        """Compute squared norm from Laguerre coefficients."""
        return np.dot(c**2, weight_vec)

    def positivity_penalty(roots):
        """
        Smooth penalty for roots not being positive and real.
        Uses soft constraints based on real parts and magnitudes.
        """
        if roots is None:
            return 1e10
        
        # Penalize complex roots (non-zero imaginary parts)
        imag_penalty = np.sum(np.abs(roots.imag))**2
        
        # Penalize non-positive real parts
        real_parts = roots.real
        neg_penalty = np.sum(np.maximum(0, -real_parts + 1e-6))**2
        
        # Penalize very small roots (to keep them well-separated from 0)
        small_penalty = np.sum(np.maximum(0, 1e-6 - real_parts))**2
        
        return imag_penalty + neg_penalty + small_penalty

    def cost_func_lag(c):
        """
        Minimize 1/norm_sq + penalty for non-positive roots.
        c: 13 Laguerre coefficients.
        """
        norm_sq = compute_norm_sq_lag(c)
        
        if norm_sq <= 1e-10:
            return 1e10
        
        p, roots = laguerre_to_roots(c)
        
        if roots is None:
            return 1e10
        
        penalty = positivity_penalty(roots)
        
        # The objective is 1/norm_sq + weighted penalty
        # We want to minimize this
        return 1.0 / norm_sq + 100.0 * penalty

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
        
        # c = A^{-1} @ p (since p = A @ c, so c = A^{-1} @ p)
        c = A_inv @ p
        
        # Norm squared: ||P||^2 = sum c_k^2 * (k+1)
        norm_sq = np.dot(c**2, weight_vec)
        
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def cost_func(roots):
        """
        Minimize the upper bound on C4.
        Bound ~ 1 / ||P||^2.
        So we minimize 1/norm_sq, which is equivalent to maximizing norm_sq.
        We return 1/norm_sq.
        """
        roots = np.asarray(roots)
        
        # Check for numerical issues
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
            
        norm_sq = compute_norm_sq(roots)
        
        if norm_sq <= 1e-10:
            return 1e10
            
        return 1.0 / norm_sq

    # Seed: geometric sequence a=4, q=2
    seed_roots = 4.0 * (2.0 ** np.arange(12))
    
    # Convert seed roots to Laguerre coefficients for initial Laguerre-space optimization
    coeffs_desc = np.poly(seed_roots)
    p_seed = coeffs_desc[::-1]
    c_seed = A_inv @ p_seed
    
    # Strategy 1: Optimize in Laguerre coefficient space
    # Bounds for Laguerre coefficients - use reasonable ranges
    # The leading coefficient should be ~1 for monic polynomial
    # Lower-order coefficients can vary more
    c_bounds = [(-100, 100)] * 13
    c_bounds[12] = (0.5, 2.0)  # Leading coefficient should be near 1
    
    best_roots = None
    best_val = 1e10
    best_c = None

    # Run differential evolution in Laguerre space
    seeds = [42, 123, 456]
    
    for seed in seeds:
        elapsed = time.time() - start_time
        if elapsed > deadline - 60:  # Leave 60s for final polish
            break
        
        try:
            result = differential_evolution(
                cost_func_lag, 
                c_bounds, 
                seed=seed, 
                maxiter=100, 
                popsize=15, 
                tol=1e-6, 
                polish=True,
                init='sobol'
            )
            
            # Convert result to roots
            p, roots = laguerre_to_roots(result.x)
            
            if roots is not None:
                # Check if roots are positive and real
                if np.all(np.abs(roots.imag) < 1e-6) and np.all(roots.real > 0):
                    real_roots = np.sort(roots.real)
                    val = cost_func(real_roots)
                    
                    if val < best_val:
                        best_val = val
                        best_roots = real_roots
                        best_c = result.x.copy()
                        
        except Exception as e:
            pass

    # Fallback: seed initialization
    seed_val = cost_func(seed_roots)
    
    if best_roots is None or seed_val < best_val:
        best_val = seed_val
        best_roots = seed_roots.copy()

    # Strategy 2: Local polish in root space (original approach)
    if time.time() - start_time < deadline - 10 and best_roots is not None:
        # Convert to log space for constrained optimization
        log_roots = np.log(best_roots)
        bounds = [(0.0, 9.5)] * 12
        
        try:
            polish_result = minimize(
                cost_func, log_roots, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_roots = np.exp(polish_result.x)
        except Exception:
            pass

    # Ensure roots are positive
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)
    
    # Absolute fallback
    return [4.0 * (2.0 ** i) for i in range(12)]
