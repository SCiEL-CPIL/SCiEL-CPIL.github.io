import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math
import warnings
warnings.filterwarnings('ignore')

def construct():
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin before 600s limit

    m = 12
    # The polynomial is P(x) = prod_{i=1}^m (x - r_i)^2
    # Degree of P is 2*m = 24.
    # We need Laguerre basis up to degree 24.
    max_deg = 2 * m
    
    # Precompute the transformation matrix from standard basis to Laguerre basis
    # P(x) = sum c_k L_k^{(1)}(x)
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    
    A = np.zeros((max_deg + 1, max_deg + 1))
    for k in range(max_deg + 1):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    # We need to solve for c given p (coeffs of P in standard basis, ascending).
    # P(x) = sum p_j x^j
    # Also P(x) = sum c_k L_k(x) = sum c_k sum A[k,j] x^j = sum_j (sum_k c_k A[k,j]) x^j
    # So p_j = sum_k c_k A[k,j].
    # In matrix form: p = A^T c  => c = (A^T)^{-1} p
    # Note: A is upper triangular (A[k,j]=0 if j>k). A^T is lower triangular.
    # Since A.T is lower triangular, use cho_solve or just inv for 25x25
    
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)
    
    # Precompute weights for norm
    weights = np.arange(max_deg + 1) + 1

    def compute_norm_sq(roots):
        """
        Compute the squared norm of the polynomial P(x) = prod (x - r_i)^2 
        in the Laguerre basis L_k^{(1)}.
        roots: array of m positive numbers.
        """
        # Build polynomial Q(x) = prod (x - r_i)
        # np.poly returns descending coeffs: [1, c1, ..., cm]
        coeffs_Q_desc = np.poly(roots)
        
        # P(x) = Q(x)^2
        # Convolve coeffs_Q_desc with itself to get P_coeffs_desc
        coeffs_P_desc = np.convolve(coeffs_Q_desc, coeffs_Q_desc)
        
        # Convert to ascending order for the matrix multiplication
        p = coeffs_P_desc[::-1]
        
        # Ensure p has length max_deg + 1
        if len(p) < max_deg + 1:
            p = np.pad(p, (0, max_deg + 1 - len(p)))
        else:
            p = p[:max_deg + 1]
            
        # c = A_T_inv @ p
        c = A_T_inv @ p
        
        # Norm squared: <L_k, L_l> = delta_kl * (k+1) for alpha=1
        # ||P||^2 = sum c_k^2 * (k+1)
        norm_sq = np.dot(c**2, weights)
        
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def cost_func(log_roots):
        """
        Minimize the upper bound on C4.
        Bound ~ 1 / ||P||^2.
        So we minimize 1/norm_sq, which is equivalent to maximizing norm_sq.
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
    # 1. Global search over 12 log-root parameters using Differential Evolution.
    # 2. Fewer restarts but more iterations per restart.
    # 3. Final high-precision polish with multiple methods.
    
    # Seed: geometric sequence a=4, q=2
    # r_i = 4 * 2^i
    seed_log_roots = np.log(4.0) + np.arange(m) * np.log(2.0)
    
    # Bounds for log_roots
    # Roots should be positive.
    # Based on seed (4 to 4*2^11 ~ 8192), log range is [~1.4, ~9].
    # Allow a wider range.
    bounds = [(-3.0, 11.0)] * m
    
    best_roots = None
    best_val = 1e10
    best_log_roots = None

    # Run Differential Evolution with fewer restarts but more iterations
    seeds = [42, 123, 456, 789, 101]
    
    for seed in seeds:
        elapsed = time.time() - start_time
        if elapsed > deadline - 90:  # Leave 90s for polish and fallback
            break
        
        try:
            # Use moderate popsize and more iterations
            result = differential_evolution(
                cost_func, 
                bounds, 
                seed=seed, 
                maxiter=200, 
                popsize=15, 
                tol=1e-12, 
                polish=True,  # L-BFGS-B polish
                init='sobol',
                update_interval=1
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x
                best_roots = np.exp(best_log_roots)
                
        except Exception as e:
            pass

    # Fallback / Initial guess if DE fails or to ensure we have a baseline
    # The seed from the prompt: r0=4, q=2
    seed_roots = 4.0 * (2.0 ** np.arange(m))
    seed_val = cost_func(seed_log_roots)
    
    if best_roots is None or seed_val < best_val:
        best_val = seed_val
        best_roots = seed_roots
        best_log_roots = seed_log_roots

    # Final high-precision polish pass with multiple methods
    if time.time() - start_time < deadline - 20 and best_log_roots is not None:
        # First: L-BFGS-B with tight tolerances
        try:
            polish_result = minimize(
                cost_func, best_log_roots, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-16, 'maxls': 50}
            )
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_log_roots = polish_result.x
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass
        
        # Second: Try Nelder-Mead for local refinement
        if time.time() - start_time < deadline - 10:
            try:
                nm_result = minimize(
                    cost_func, best_log_roots, method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
                )
                if nm_result.fun < best_val:
                    best_val = nm_result.fun
                    best_log_roots = nm_result.x
                    best_roots = np.exp(best_log_roots)
            except Exception:
                pass
        
        # Third: Final L-BFGS-B polish after Nelder-Mead
        if time.time() - start_time < deadline - 5:
            try:
                final_result = minimize(
                    cost_func, best_log_roots, method='L-BFGS-B',
                    bounds=bounds, options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-16}
                )
                if final_result.fun < best_val:
                    best_val = final_result.fun
                    best_log_roots = final_result.x
                    best_roots = np.exp(best_log_roots)
            except Exception:
                pass

    # Ensure roots are positive and sorted (though order doesn't matter for the product)
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)
    
    # Absolute fallback
    return [4.0 * (2.0 ** i) for i in range(m)]
