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

    def cost_func(log_vars):
        """
        Minimize the upper bound on C4.
        The 12 roots are represented as squares of 6 positive variables:
        r_i = v_i^2 for i = 0, ..., 5, and the 12 roots are [v_0^2, v_1^2, ..., v_5^2, v_0^2, v_1^2, ..., v_5^2]
        
        Wait, the task says "12 double roots". This means each root appears with multiplicity 2.
        So we have 6 distinct roots, each appearing twice.
        Let the distinct roots be d_0, ..., d_5.
        The 12 roots are [d_0, d_0, d_1, d_1, ..., d_5, d_5].
        
        The strategy says: "Represent the 12 double roots as the squared values of 6 positive variables to structurally enforce the double-root multiplicity and positivity"
        
        This interpretation is slightly ambiguous. Does it mean:
        1. The 12 roots are 6 pairs of identical roots? (i.e., 6 distinct values, each with multiplicity 2)
        2. Or does it mean we have 12 roots, and we represent them as squares of 6 variables in some other way?
        
        Given "12 double roots", it most likely means 6 distinct roots, each of multiplicity 2.
        So P(x) = prod_{i=0}^{5} (x - d_i)^2.
        
        Let d_i = v_i^2 where v_i > 0.
        Then P(x) = prod_{i=0}^{5} (x - v_i^2)^2.
        
        We optimize over log(v_i) for i=0..5.
        """
        # log_vars is an array of 6 values: log(v_0), ..., log(v_5)
        v = np.exp(log_vars)
        d = v ** 2  # The 6 distinct roots
        
        # The 12 roots (with multiplicity)
        roots = np.concatenate([d, d])
        
        # Check for numerical issues
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
            
        norm_sq = compute_norm_sq(roots)
        
        if norm_sq <= 1e-10:
            return 1e10
            
        return 1.0 / norm_sq

    # Optimization Strategy:
    # 1. Global search over 6 log-variable parameters using Differential Evolution.
    # 2. Multiple restarts to avoid local minima.
    # 3. Final polish with L-BFGS-B.
    
    # Seed: geometric sequence for the 6 distinct roots
    # Let d_i = 4 * 2^(2i) = 4 * 4^i for i=0..5
    # Then v_i = sqrt(d_i) = 2 * 2^i = 2^(i+1)
    # log(v_i) = (i+1) * log(2)
    seed_log_vars = (np.arange(6) + 1) * np.log(2.0)
    
    # Bounds for log_vars
    # d_i = v_i^2. If d_i ranges from ~0.01 to ~10000, then v_i ranges from ~0.1 to ~100.
    # log(v_i) ranges from ~-2.3 to ~4.6.
    # Let's allow a wider range: log(0.01) ~ -4.6 to log(1000) ~ 6.9 for v_i.
    bounds = [(-4.0, 7.0)] * 6

    best_vars = None
    best_val = 1e10
    best_log_vars = None

    # Run Differential Evolution with multiple restarts
    # We have ~590s. DE is stochastic.
    # With 6D instead of 12D, DE should be much faster.
    
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808]
    
    for seed in seeds:
        elapsed = time.time() - start_time
        if elapsed > deadline - 60:  # Leave 60s for polish and fallback
            break
        
        try:
            # Use smaller popsize and maxiter for speed, relying on multiple restarts
            # With 6D, we can afford larger populations and more iterations.
            result = differential_evolution(
                cost_func, 
                bounds, 
                seed=seed, 
                maxiter=300, 
                popsize=25, 
                tol=1e-10, 
                polish=True,  # L-BFGS-B polish
                init='sobol' # Better initial distribution
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_log_vars = result.x
                best_vars = np.exp(best_log_vars)
                
        except Exception as e:
            pass

    # Fallback / Initial guess if DE fails or to ensure we have a baseline
    # The seed: v_i = 2^(i+1), d_i = 4^i * 4
    seed_vars = 2.0 ** (np.arange(6) + 1)
    seed_val = cost_func(seed_log_vars)
    
    if best_vars is None or seed_val < best_val:
        best_val = seed_val
        best_vars = seed_vars
        best_log_vars = seed_log_vars

    # Final high-precision polish pass
    if time.time() - start_time < deadline - 10 and best_log_vars is not None:
        try:
            polish_result = minimize(
                cost_func, best_log_vars, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_log_vars = polish_result.x
                best_vars = np.exp(best_log_vars)
        except Exception:
            pass

    # Construct the final 12 roots (6 distinct roots, each repeated twice)
    if best_vars is not None:
        d = best_vars ** 2
        d = np.maximum(d, 1e-6)
        # The 12 roots: d_0, d_0, d_1, d_1, ..., d_5, d_5
        roots_12 = np.concatenate([d, d])
        return list(roots_12)
    
    # Absolute fallback
    # Use the seed: d_i = 4 * 4^i = 4^(i+1)
    d_seed = 4.0 ** (np.arange(6) + 1)
    roots_12 = np.concatenate([d_seed, d_seed])
    return list(roots_12)
