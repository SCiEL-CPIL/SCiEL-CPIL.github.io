import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 580  # 50s safety margin before 630s kill

    m = 12

    # Precompute the transformation matrix from monomial basis to Laguerre basis
    # L_k^(1)(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    A = np.zeros((13, 13))
    for k in range(13):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    # c = A^T^{-1} @ p, where p are monomial coefficients (ascending)
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

    weights = np.arange(13) + 1  # (k+1) for k=0..12

    def norm_sq_from_roots(roots):
        """Compute squared norm from 12 positive roots."""
        # Build polynomial in ascending coefficients: P(x) = prod(x - r_i)
        # Use iterative multiplication for numerical stability
        p = np.array([1.0])
        for r in roots:
            # p_new = p * (x - r) = [p[0]*(-r), p[0]*1 + p[1]*(-r), ..., p[-1]*1]
            # In ascending order: multiply by (x - r)
            p_new = np.zeros(len(p) + 1)
            p_new[0] = -r * p[0]
            for i in range(1, len(p)):
                p_new[i] = p[i-1] - r * p[i]
            p_new[len(p)] = p[-1]
            p = p_new
        
        # p should have length 13 (degree 12)
        if len(p) != 13:
            return 0.0
        
        c = A_T_inv @ p
        norm_sq = np.sum(c**2 * weights)
        
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def cost_func(log_roots):
        """Minimize 1/norm_sq (equivalent to maximizing norm_sq)."""
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            return 1e10
        
        roots = np.exp(log_roots)
        
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
        
        ns = norm_sq_from_roots(roots)
        
        if ns <= 1e-10:
            return 1e10
        
        return 1.0 / ns

    # Generate multiple starting points
    starting_points = []
    
    # 1. Original seed: geometric sequence r_i = 4 * 2^i
    seed_roots = 4.0 * (2.0 ** np.arange(12))
    starting_points.append(np.log(seed_roots))
    
    # 2. Perturbed seeds with different bases
    for base in [2.0, 3.0, 5.0, 8.0, 16.0]:
        for ratio in [1.5, 1.8, 2.0, 2.2, 2.5, 3.0]:
            roots = base * (ratio ** np.arange(12))
            if np.all(roots > 0) and np.all(np.isfinite(roots)):
                starting_points.append(np.log(roots))
    
    # 3. Random starting points in log-space
    rng = np.random.RandomState(42)
    for _ in range(20):
        log_r = rng.uniform(-2.0, 10.0, size=12)
        starting_points.append(log_r)
    
    # 4. Sorted random points
    for _ in range(10):
        log_r = np.sort(rng.uniform(-2.0, 10.0, size=12))
        starting_points.append(log_r)

    best_val = 1e10
    best_log_roots = None
    best_roots = None

    # Phase 1: Local optimization from each starting point using L-BFGS-B
    bounds = [(-4.0, 12.0)] * 12

    for i, log_r0 in enumerate(starting_points):
        elapsed = time.time() - start_time
        if elapsed > deadline - 60:
            break
        
        try:
            result = minimize(
                cost_func, log_r0, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x.copy()
                best_roots = np.exp(best_log_roots).copy()
        except Exception:
            pass

    # Phase 2: If we have a good result, do a final high-precision polish
    if best_log_roots is not None and time.time() - start_time < deadline - 20:
        try:
            polish_result = minimize(
                cost_func, best_log_roots, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-16}
            )
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_log_roots = polish_result.x.copy()
                best_roots = np.exp(best_log_roots).copy()
        except Exception:
            pass

    # Phase 3: Use DE if we still have time and haven't found a great result
    if time.time() - start_time < deadline - 120 and (best_val > 30.0 or best_roots is None):
        try:
            de_result = differential_evolution(
                cost_func,
                bounds,
                seed=12345,
                maxiter=150,
                popsize=25,
                tol=1e-12,
                polish=True,
                init='sobol'
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_log_roots = de_result.x.copy()
                best_roots = np.exp(best_log_roots).copy()
        except Exception:
            pass

    # Ensure we have a valid result
    if best_roots is None:
        best_roots = seed_roots
        best_val = cost_func(np.log(seed_roots))

    # Final sanity check: ensure positive roots
    best_roots = np.maximum(best_roots, 1e-8)
    
    return list(best_roots)
