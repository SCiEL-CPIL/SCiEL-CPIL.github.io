"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import math

def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Returns a list of 12 positive root locations.
    Score is the upper bound on C4 (lower is better for the evaluator's raw value,
    but the search score is seed/value, so higher search score = better).
    """
    start_time = time.time()
    DEADLINE = 630.0  # seconds wall-clock limit
    SAFETY_MARGIN = 20.0  # return 20s before the kill

    def time_ok():
        return (time.time() - start_time) < (DEADLINE - SAFETY_MARGIN)

    m = 12
    n = 1

    # Precompute the transformation matrix from standard basis to Laguerre basis
    # P(x) = sum c_k L_k^{(1)}(x)
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    
    A = np.zeros((13, 13))  # A[k, j] = coeff of x^j in L_k^{(1)}(x)
    for k in range(13):
        for j in range(k + 1):
            # Use math.comb for exact integer arithmetic where possible, then convert
            # Note: math.comb returns int, so we divide by factorial
            try:
                A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
            except OverflowError:
                # Fallback for very large numbers, though m=12 is small
                A[k, j] = ((-1)**j) * (math.factorial(k+1) / (math.factorial(j+1) * math.factorial(k-j))) / math.factorial(j)
    
    # Solve for c given p. c = (A^T)^{-1} p
    # A is lower triangular (L_k has degree k). A[k, j] is 0 if j > k.
    # We want c such that p = A^T c ? No.
    # L_k(x) = sum_j A[k,j] x^j.
    # Q(x) = sum_k c_k L_k(x) = sum_k c_k sum_j A[k,j] x^j = sum_j (sum_k c_k A[k,j]) x^j.
    # So p_j = sum_k c_k A[k,j].
    # In matrix form: p = A^T c.
    # So c = (A^T)^{-1} p.
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

    def compute_norm_sq(roots):
        """
        Compute the squared norm of the polynomial Q(x) = prod(x - r_i) in the Laguerre basis.
        roots: array of 12 positive numbers.
        """
        # Build polynomial coefficients in standard basis (ascending)
        # np.poly returns descending coeffs: [1, c1, ..., cm]
        # We need ascending: [c0, c1, ..., cm]
        try:
            coeffs_desc = np.poly(roots)
            p = coeffs_desc[::-1] # ascending
        except Exception:
            return 0.0
        
        # Ensure p is length 13 (degree 12)
        if len(p) < 13:
            p = np.pad(p, (0, 13 - len(p)))
        else:
            p = p[:13]
            
        # c = A_T_inv @ p
        c = A_T_inv @ p
        
        # Norm squared: <L_k, L_l> = delta_kl * (k+1) for alpha=1
        # ||Q||^2 = sum c_k^2 * (k+1)
        norm_sq = np.sum(c**2 * (np.arange(13) + 1))
        
        if norm_sq <= 1e-10 or not np.isfinite(norm_sq):
            return 0.0
        return norm_sq

    def cost_func(log_roots):
        """
        Minimize the upper bound on C4.
        Bound ~ 1 / ||Q||^2.
        So we minimize 1/norm_sq, which is equivalent to maximizing norm_sq.
        We return 1/norm_sq.
        """
        if np.any(np.isnan(log_roots)) or np.any(np.isinf(log_roots)):
            return 1e10
            
        # Clip to prevent overflow in exp
        log_roots_clipped = np.clip(log_roots, -10, 30)
        roots = np.exp(log_roots_clipped)
        
        # Check for numerical issues
        if np.any(roots <= 0) or np.any(roots > 1e12):
            return 1e10
            
        norm_sq = compute_norm_sq(roots)
        
        if norm_sq <= 1e-10:
            return 1e10
            
        return 1.0 / norm_sq

    # Optimization Strategy:
    # 1. Global search over 12 log-root parameters using Differential Evolution.
    # 2. Multiple restarts to avoid local minima.
    # 3. Final polish with L-BFGS-B.
    
    # Bounds for log_roots
    # Roots should be positive.
    # Geometric sequences like 4*2^i give logs in [1.4, 9.2].
    # Allow a wider range: log(0.01) ~ -4.6 to log(10000) ~ 9.2
    bounds = [(-4.0, 12.0)] * 12
    
    best_roots = None
    best_val = 1e10
    best_log_roots = None

    # Run Differential Evolution with multiple restarts
    # We have ~590s. DE is stochastic.
    # Let's run a few independent DEs with different seeds and settings.
    
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
    
    for seed in seeds:
        if not time_ok():
            break
        
        try:
            # Use smaller popsize and maxiter for speed, relying on multiple restarts
            # popsize=15, maxiter=100 is usually fast for 12D.
            # With the fast cost function, we can afford more iterations.
            result = differential_evolution(
                cost_func, 
                bounds, 
                seed=seed, 
                maxiter=300, 
                popsize=20, 
                tol=1e-12, 
                polish=True,  # L-BFGS-B polish
                init='sobol' # Better initial distribution
            )
            
            if result.fun < best_val and np.all(np.isfinite(result.x)):
                best_val = result.fun
                best_log_roots = result.x.copy()
                best_roots = np.exp(np.clip(best_log_roots, -10, 30))
                
        except Exception:
            pass

    # Fallback / Initial guess if DE fails or to ensure we have a baseline
    # The seed from the prompt: r0=4, q=2
    seed_roots = 4.0 * (2.0 ** np.arange(12))
    seed_log_roots = np.log(seed_roots)
    seed_val = cost_func(seed_log_roots)
    
    if best_roots is None or seed_val < best_val:
        best_val = seed_val
        best_roots = seed_roots
        best_log_roots = seed_log_roots

    # Final high-precision polish pass
    if time_ok() and best_log_roots is not None:
        try:
            polish_result = minimize(
                cost_func, best_log_roots, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if polish_result.fun < best_val and np.all(np.isfinite(polish_result.x)):
                best_val = polish_result.fun
                best_log_roots = polish_result.x.copy()
                best_roots = np.exp(np.clip(best_log_roots, -10, 30))
        except Exception:
            pass

    # Additional local search hops to escape potential local minima
    if time_ok() and best_log_roots is not None:
        rng = np.random.default_rng(12345)
        n_hops = 50
        for hop in range(n_hops):
            if not time_ok():
                break
            # Perturb the best solution
            perturbation = rng.normal(0, 0.5, size=12)
            perturbed_log = best_log_roots + perturbation
            try:
                res = minimize(cost_func, perturbed_log, method='L-BFGS-B',
                               bounds=bounds, options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
                if res.fun < best_val and np.all(np.isfinite(res.x)):
                    best_val = res.fun
                    best_log_roots = res.x.copy()
                    best_roots = np.exp(np.clip(best_log_roots, -10, 30))
            except Exception:
                continue

    # Ensure roots are positive and sorted (though order doesn't matter for the product)
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        best_roots = np.sort(best_roots)
        return list(best_roots)
    
    # Absolute fallback
    return [4.0 * (2.0 ** i) for i in range(12)]
# EVOLVE-BLOCK-END
