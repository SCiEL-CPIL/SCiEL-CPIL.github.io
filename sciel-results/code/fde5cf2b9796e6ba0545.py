import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 580

    m = 12

    # Precompute the transformation matrix from monomial basis to Laguerre basis
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    
    A = np.zeros((m + 1, m + 1))
    for k in range(m + 1):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    # Solve for c given p. c = (A^T)^{-1} p
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)
    
    # Precompute the weight vector for norm calculation
    weight_vec = np.arange(m + 1) + 1

    def compute_norm_sq(roots):
        """
        Compute the squared norm of the polynomial in the Laguerre basis.
        roots: array of 12 positive numbers.
        """
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1]
        c = A_T_inv @ p
        norm_sq = np.dot(c**2, weight_vec)
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def cost_func(log_roots):
        """
        Minimize the upper bound on C4 = 1 / ||P||^2
        """
        roots = np.exp(log_roots)
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
        norm_sq = compute_norm_sq(roots)
        if norm_sq <= 1e-10:
            return 1e10
        return 1.0 / norm_sq

    # Multiple initializations: geometric sequences with different ratios
    def make_geo_init(a, q, n=12):
        return np.log(a) + np.arange(n) * np.log(q)
    
    # Several geometric initializations
    inits = []
    for a in [1.5, 2.0, 3.0, 4.0, 5.0]:
        for q in [1.5, 1.8, 2.0, 2.2, 2.5]:
            init = make_geo_init(a, q)
            if np.all(np.isfinite(init)) and np.all(init > 0):
                inits.append(init)
    
    # Also add the original seed
    inits.append(make_geo_init(4.0, 2.0))
    
    # Add some random perturbations of the best geometric sequences
    rng = np.random.RandomState(42)
    for _ in range(20):
        base = rng.choice(len(inits))
        perturbed = inits[base] + rng.randn(12) * 0.3
        if np.all(perturbed > 0):
            inits.append(perturbed)
    
    # Bounds
    bounds = [(0.0, 10.0)] * 12

    best_roots = None
    best_val = 1e10
    best_log_roots = None

    # Phase 1: Local optimization from multiple starting points
    for i, init in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > deadline - 120:  # Leave 120s for DE and final polish
            break
        
        try:
            result = minimize(
                cost_func, init, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass
    
    # Phase 2: Differential evolution for global exploration
    seeds = [42, 123, 456, 789, 101, 202, 303]
    
    for seed in seeds:
        elapsed = time.time() - start_time
        if elapsed > deadline - 60:  # Leave 60s for final polish
            break
        
        try:
            result = differential_evolution(
                cost_func, 
                bounds, 
                seed=seed, 
                maxiter=80, 
                popsize=12, 
                tol=1e-10, 
                polish=True,
                init='sobol'
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x
                best_roots = np.exp(best_log_roots)
                
        except Exception:
            pass

    # Fallback: seed initialization
    seed_roots = 4.0 * (2.0 ** np.arange(12))
    seed_log_roots = np.log(seed_roots)
    seed_val = cost_func(seed_log_roots)
    
    if best_roots is None or seed_val < best_val:
        best_val = seed_val
        best_roots = seed_roots
        best_log_roots = seed_log_roots

    # Phase 3: Final high-precision polish
    if time.time() - start_time < deadline - 5 and best_log_roots is not None:
        try:
            polish_result = minimize(
                cost_func, best_log_roots, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-16}
            )
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_log_roots = polish_result.x
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass

    # Ensure roots are positive
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)
    
    return [4.0 * (2.0 ** i) for i in range(12)]
