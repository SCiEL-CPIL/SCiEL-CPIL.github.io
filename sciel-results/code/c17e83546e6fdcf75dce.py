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
        We return -log(norm_sq) to improve numerical stability and gradient scaling.
        """
        roots = np.exp(log_roots)
        
        # Check for numerical issues
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
            
        norm_sq = compute_norm_sq(roots)
        
        if norm_sq <= 1e-10:
            return 1e10
            
        # Minimizing -log(norm_sq) is equivalent to maximizing norm_sq
        return -np.log(norm_sq)

    # Optimization Strategy:
    # 1. Global search over 12 log-root parameters using Differential Evolution.
    # 2. Multiple restarts to avoid local minima.
    # 3. Final polish with L-BFGS-B and Powell.
    
    # Seed: geometric sequence a=4, q=2
    # r_i = 4 * 2^i
    # log_r_i = log(4) + i * log(2)
    seed_log_roots = np.log(4.0) + np.arange(12) * np.log(2.0)
    
    # Bounds for log_roots
    # Roots should be positive.
    # If roots are too small or too large, the norm might behave badly.
    # Based on seed (4 to 4*2^11 ~ 8192), log range is [~1.4, ~9].
    # Let's allow a wider range: log(0.01) ~ -4.6 to log(10000) ~ 9.2
    bounds = [(-4.0, 10.0)] * 12
    
    best_roots = None
    best_val = 1e10
    best_log_roots = None

    # Run Differential Evolution with multiple restarts
    # We have ~590s. DE is stochastic.
    # Let's run a few independent DEs with different seeds and settings.
    
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808]
    
    for seed in seeds:
        elapsed = time.time() - start_time
        if elapsed > deadline - 60:  # Leave 60s for polish and fallback
            break
        
        try:
            # Use smaller popsize and maxiter for speed, relying on multiple restarts
            # popsize=15, maxiter=100 is usually fast for 12D.
            result = differential_evolution(
                cost_func, 
                bounds, 
                seed=seed, 
                maxiter=300, 
                popsize=25, 
                tol=1e-12, 
                polish=True,  # L-BFGS-B polish
                init='sobol' # Better initial distribution
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x
                best_roots = np.exp(best_log_roots)
                
        except Exception as e:
            pass

    # Fallback / Initial guess if DE fails or to ensure we have a baseline
    # The seed from the prompt: r0=4, q=2
    seed_roots = 4.0 * (2.0 ** np.arange(12))
    seed_val = cost_func(seed_log_roots)
    
    if best_roots is None or seed_val < best_val:
        best_val = seed_val
        best_roots = seed_roots
        best_log_roots = seed_log_roots

    # Final high-precision polish pass
    if time.time() - start_time < deadline - 10 and best_log_roots is not None:
        try:
            # L-BFGS-B polish
            polish_result = minimize(
                cost_func, best_log_roots, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_log_roots = polish_result.x
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass
            
        # Powell polish to break out of any local minima or refine without gradients
        if time.time() - start_time < deadline - 5 and best_log_roots is not None:
            try:
                powell_result = minimize(
                    cost_func, best_log_roots, method='Powell',
                    bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12}
                )
                if powell_result.fun < best_val:
                    best_val = powell_result.fun
                    best_log_roots = powell_result.x
                    best_roots = np.exp(best_log_roots)
            except Exception:
                pass

    # Ensure roots are positive and sorted (though order doesn't matter for the product)
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)
    
    # Absolute fallback
    return [4.0 * (2.0 ** i) for i in range(12)]
