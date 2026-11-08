import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin before 630s kill

    m = 12
    alpha = 1

    # Precompute the transformation matrix from monomial basis to Laguerre basis
    # L_k^(alpha)(x) = sum_{j=0}^k A[k,j] x^j
    # For alpha=1: A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    A = np.zeros((m + 1, m + 1))
    for k in range(m + 1):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k + alpha, j + 1) / math.factorial(j)
    
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
        
        # Norm squared: <L_k, L_l> = delta_kl * (k+alpha) for alpha=1
        # ||P||^2 = sum c_k^2 * (k+1)
        weights = np.arange(m + 1) + alpha
        norm_sq = np.sum(c**2 * weights)
        
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def cost_func(log_roots):
        """
        Minimize the upper bound on C4 = 1/||P||^2.
        """
        roots = np.exp(log_roots)
        
        # Check for numerical issues
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
            
        norm_sq = compute_norm_sq(roots)
        
        if norm_sq <= 1e-10:
            return 1e10
            
        return 1.0 / norm_sq

    # Define bounds for log_roots
    # Roots in [0.01, 1000] -> logs in [-4.6, 6.9]
    # Widen slightly: [1e-4, 1e4] -> [-9.2, 9.2]
    log_low = -8.0
    log_high = 10.0
    bounds = [(log_low, log_high)] * 12

    # Generate diverse initial guesses
    rng = np.random.default_rng(42)
    
    initial_guesses = []
    
    # 1. Original geometric seed
    seed_log_roots = np.log(4.0) + np.arange(12) * np.log(2.0)
    initial_guesses.append(seed_log_roots)
    
    # 2. More moderate geometric sequences
    for base in [1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0]:
        for start in [0.5, 1.0, 2.0, 4.0, 8.0]:
            logs = np.log(start) + np.arange(12) * np.log(base)
            if np.all(logs > log_low) and np.all(logs < log_high):
                initial_guesses.append(logs)
    
    # 3. Linear spacing in log space
    for center in [0.0, 1.0, 2.0, 3.0]:
        for width in [2.0, 4.0, 6.0, 8.0]:
            logs = center + width * (np.arange(12) - 5.5) / 5.5
            if np.all(logs > log_low) and np.all(logs < log_high):
                initial_guesses.append(logs)
    
    # 4. Random log-uniform starts
    for _ in range(200):
        logs = rng.uniform(log_low, log_high, size=12)
        initial_guesses.append(logs)
    
    # 5. Clustered roots (some close together)
    for _ in range(50):
        # Pick a center and spread
        center = rng.uniform(-2, 5)
        spread = rng.uniform(0.5, 4)
        logs = center + spread * rng.normal(0, 1, size=12)
        logs = np.clip(logs, log_low, log_high)
        initial_guesses.append(logs)

    best_val = 1e10
    best_log_roots = None
    best_roots = None

    # Run L-BFGS-B on each guess with time management
    for i, init in enumerate(initial_guesses):
        elapsed = time.time() - start_time
        if elapsed > deadline - 15:
            break
            
        try:
            result = minimize(
                cost_func, init, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x.copy()
                best_roots = np.exp(best_log_roots)
                
        except Exception:
            pass

    # If we have time, try a few DE runs to escape potential local minima
    if time.time() - start_time < deadline - 60:
        try:
            result_de = differential_evolution(
                cost_func, 
                bounds, 
                seed=123, 
                maxiter=100, 
                popsize=15, 
                tol=1e-10, 
                polish=True,
                init='sobol'
            )
            if result_de.fun < best_val:
                best_val = result_de.fun
                best_log_roots = result_de.x.copy()
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass

    # Final high-precision polish
    if time.time() - start_time < deadline - 5 and best_log_roots is not None:
        try:
            polish_result = minimize(
                cost_func, best_log_roots, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-18}
            )
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_log_roots = polish_result.x.copy()
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass

    # Ensure roots are positive
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-10)
        return list(best_roots)
    
    # Absolute fallback
    return [4.0 * (2.0 ** i) for i in range(12)]
