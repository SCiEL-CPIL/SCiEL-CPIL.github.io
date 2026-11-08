import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math

def construct():
    # (placeholder: function body follows below)
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin before 600s limit

    m = 12
    max_deg = 2 * m

    # Build the conversion from monic standard basis x^j to generalized Laguerre L_k^(alpha)(x).
    # We need the exact convention used by the Laguerre inner product.
    # The standard (physics) Laguerre polynomials L_k^(alpha) have orthogonality:
    # ∫_0^∞ e^{-x} x^alpha L_k^(alpha)(x) L_l^(alpha)(x) dx = (k+alpha)!/k! δ_kl
    # For alpha=1: weight = (k+1)!/k! = k+1.
    #
    # The monic Laguerre polynomial is:
    #   \tilde{L}_k^(alpha)(x) = (-1)^k k! L_k^(alpha)(x)
    # and the standard basis x^j can be expanded in monic Laguerre basis.
    #
    # We want: P(x) = sum_{k=0}^{max_deg} d_k \tilde{L}_k^(1)(x)
    # Then the norm squared is sum_k d_k^2 * (k+1)^2 / (k!)^2 * (k+1)!/k! = sum_k d_k^2 * (k+1)^3 / (k!)^2
    # Wait, let me recompute.
    #
    # Actually, the inner product of monic Laguerre polynomials:
    # ∫ e^{-x} x^1 \tilde{L}_k^(1)(x) \tilde{L}_l^(1)(x) dx = (k!)^2 * (k+1)!/k! = (k!)^2 * (k+1)
    # So ||\tilde{L}_k||^2 = (k!)^2 * (k+1)
    #
    # If P = sum d_k \tilde{L}_k, then ||P||^2 = sum d_k^2 * (k!)^2 * (k+1)
    #
    # Alternatively, work with standard (non-monic) Laguerre:
    # P = sum c_k L_k^(1)(x), then ||P||^2 = sum c_k^2 * (k+1)
    # And \tilde{L}_k = (-1)^k k! L_k, so d_k = (-1)^k k! c_k
    # Thus c_k = (-1)^k d_k / k!
    # And ||P||^2 = sum (d_k^2 / (k!)^2) * (k+1)
    #
    # Let me use the standard Laguerre expansion directly.
    #
    # The standard Laguerre polynomial L_k^(alpha)(x) = sum_{j=0}^k (-1)^j * binom(k+alpha, k-j) / j! * x^j
    # For alpha=1: L_k^(1)(x) = sum_{j=0}^k (-1)^j * binom(k+1, k-j) / j! * x^j
    #              = sum_{j=0}^k (-1)^j * binom(k+1, j+1) / j! * x^j
    #
    # So if P(x) = sum_{j=0}^{max_deg} p_j x^j = sum_{k=0}^{max_deg} c_k L_k^(1)(x),
    # then p_j = sum_{k=j}^{max_deg} c_k * (-1)^j * binom(k+1, j+1) / j!
    #
    # This is a triangular system. Let M[k,j] = (-1)^j * binom(k+1, j+1) / j! for j <= k, else 0.
    # Then p = M^T c, so c = (M^T)^{-1} p.
    #
    # The norm squared is sum c_k^2 * (k+1).

    # Precompute M^T inverse
    M = np.zeros((max_deg + 1, max_deg + 1))
    for k in range(max_deg + 1):
        for j in range(k + 1):
            M[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    try:
        M_T_inv = np.linalg.inv(M.T)
    except np.linalg.LinAlgError:
        M_T_inv = np.linalg.pinv(M.T)
    
    weights = np.arange(max_deg + 1) + 1  # weights for ||P||^2 = sum c_k^2 * (k+1)

    def compute_norm_sq(roots):
        """
        Compute the squared norm of P(x) = prod (x - r_i)^2 in the Laguerre L_k^(1) basis.
        """
        # Build monic polynomial Q(x) = prod (x - r_i)
        # np.poly returns descending coefficients
        coeffs_Q_desc = np.poly(roots)
        
        # P(x) = Q(x)^2
        coeffs_P_desc = np.convolve(coeffs_Q_desc, coeffs_Q_desc)
        
        # Convert to ascending order
        p = coeffs_P_desc[::-1]
        
        # Pad if necessary
        if len(p) < max_deg + 1:
            p = np.pad(p, (0, max_deg + 1 - len(p)))
        else:
            p = p[:max_deg + 1]
            
        # Solve for Laguerre coefficients
        c = M_T_inv @ p
        
        # Norm squared
        norm_sq = np.sum(c**2 * weights)
        
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def cost_func(log_roots):
        """
        Minimize 1/||P||^2, which is equivalent to maximizing ||P||^2.
        """
        roots = np.exp(log_roots)
        
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
            
        norm_sq = compute_norm_sq(roots)
        
        if norm_sq <= 1e-10:
            return 1e10
            
        return 1.0 / norm_sq

    # Optimization Strategy:
    # 1. Global search over 12 log-root parameters using Differential Evolution.
    # 2. Multiple restarts to avoid local minima.
    # 3. Final high-precision polish with L-BFGS-B.
    
    # Seed: geometric sequence a=4, q=2
    seed_log_roots = np.log(4.0) + np.arange(m) * np.log(2.0)
    
    # Bounds for log_roots
    bounds = [(-4.0, 12.0)] * m
    
    best_roots = None
    best_val = 1e10
    best_log_roots = None

    # Run Differential Evolution with multiple restarts
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
    
    for seed in seeds:
        elapsed = time.time() - start_time
        if elapsed > deadline - 60:
            break
        
        try:
            result = differential_evolution(
                cost_func, 
                bounds, 
                seed=seed, 
                maxiter=150, 
                popsize=20, 
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

    # Fallback / Initial guess
    seed_roots = 4.0 * (2.0 ** np.arange(m))
    seed_val = cost_func(seed_log_roots)
    
    if best_roots is None or seed_val < best_val:
        best_val = seed_val
        best_roots = seed_roots
        best_log_roots = seed_log_roots

    # Final high-precision polish pass with tighter tolerances
    if time.time() - start_time < deadline - 10 and best_log_roots is not None:
        try:
            polish_result = minimize(
                cost_func, best_log_roots, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-18, 'maxls': 200}
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
    
    # Absolute fallback
    return [4.0 * (2.0 ** i) for i in range(m)]
