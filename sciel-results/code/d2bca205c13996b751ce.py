import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math

def construct():
    start_time = time.time()
    # Hard deadline to ensure return well before 630s kill limit
    deadline = start_time + 580 

    m = 12
    n = 1

    # Precompute the transformation matrix from standard basis to Laguerre basis
    # P(x) = sum c_k L_k^{(1)}(x)
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    
    A = np.zeros((13, 13))
    for k in range(13):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    # Solve for c given p. c = (A^T)^{-1} p
    # A is lower triangular. We need inverse of A.T which is upper triangular.
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
        So we minimize 1/norm_sq.
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
    # 1. Global search using Differential Evolution with tighter bounds.
    # 2. Local polish using L-BFGS-B.
    # 3. Multi-start local optimization as a backup/refinement.
    
    # Bounds for log_roots
    # Seed roots: 4, 8, ..., 4096. Logs: ~1.4 to ~8.3.
    # Tighten bounds to help DE converge: [0, 10] corresponds to [1, 22000].
    # This is wide enough to find better structures but narrow enough for DE to be effective.
    log_low = -1.0
    log_high = 10.0
    bounds = [(log_low, log_high)] * 12

    best_val = 1e10
    best_log_roots = None
    best_roots = None

    # Initial guess: The geometric sequence from the seed (4 * 2^i)
    seed_roots = 4.0 * (2.0 ** np.arange(12))
    seed_log_roots = np.log(seed_roots)
    seed_log_roots = np.clip(seed_log_roots, log_low, log_high)
    
    # Evaluate seed
    seed_val = cost_func(seed_log_roots)
    if seed_val < best_val:
        best_val = seed_val
        best_log_roots = seed_log_roots.copy()
        best_roots = np.exp(seed_log_roots)

    # Run Differential Evolution
    # We use a moderate popsize and maxiter to balance exploration and exploitation.
    # 'sobol' initialization provides better coverage than 'latinhypercube' for some problems.
    
    try:
        # We allow DE to run, but we will check time after.
        # polish=True ensures a local minimization step at the end of DE.
        de_result = differential_evolution(
            cost_func, 
            bounds, 
            seed=42, 
            maxiter=150, 
            popsize=25, 
            tol=1e-12, 
            polish=True,
            init='sobol',
            disp=False
        )
        
        if de_result.fun < best_val:
            best_val = de_result.fun
            best_log_roots = de_result.x.copy()
            best_roots = np.exp(best_log_roots)
            
    except Exception as e:
        pass

    # If time remains, perform additional local polishing from the best point found
    if time.time() < deadline - 5 and best_log_roots is not None:
        try:
            # High precision polish
            res = minimize(
                cost_func, 
                best_log_roots, 
                method='L-BFGS-B',
                bounds=bounds, 
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            
            if res.fun < best_val:
                best_val = res.fun
                best_log_roots = res.x.copy()
                best_roots = np.exp(res.x)
                
        except Exception:
            pass

    # If we still have time, try a few random local starts to see if we missed a basin
    # that DE didn't explore well or if the polish got stuck.
    if time.time() < deadline - 10 and best_log_roots is not None:
        rng = np.random.default_rng(123)
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            # Perturb the best solution
            perturbed = best_log_roots + rng.normal(0, 0.5, size=12)
            perturbed = np.clip(perturbed, log_low, log_high)
            
            try:
                res = minimize(
                    cost_func, 
                    perturbed, 
                    method='L-BFGS-B',
                    bounds=bounds, 
                    options={'maxiter': 1000, 'ftol': 1e-12}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_log_roots = res.x.copy()
                    best_roots = np.exp(res.x)
            except Exception:
                pass

    # Final check and return
    if best_roots is not None:
        # Ensure roots are positive
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)
    
    # Absolute fallback
    return [4.0 * (2.0 ** i) for i in range(12)]
