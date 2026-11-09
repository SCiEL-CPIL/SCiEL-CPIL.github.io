import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 590

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
    
    # Precompute the kernel matrix K such that norm_sq = p^T K p
    # norm_sq = c^T D c where D = diag(1, 2, ..., 13)
    # c = A^{-T} p  (since p = A^T c)
    # norm_sq = p^T (A^{-T})^T D A^{-T} p = p^T A^{-1} D A^{-T} p
    # So K = A^{-1} D A^{-T}
    
    D = np.diag(np.arange(1, 14))
    try:
        A_inv = np.linalg.inv(A)
        K = A_inv @ D @ A_inv.T
    except np.linalg.LinAlgError:
        A_pinv = np.linalg.pinv(A)
        K = A_pinv @ D @ A_pinv.T

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
        
        # norm_sq = p @ K @ p
        norm_sq = float(p @ K @ p)
        
        if norm_sq <= 1e-10 or np.isnan(norm_sq) or np.isinf(norm_sq):
            return 0.0
        return norm_sq

    def cost_func(log_roots):
        """
        Minimize the upper bound on C4.
        Bound ~ 1 / ||P||^2.
        So we minimize 1/norm_sq, which is equivalent to maximizing norm_sq.
        We return 1/norm_sq.
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
    # 1. Define bounds for log_roots
    # Roots in [1e-3, 1e4] -> logs in [-6.9, 9.2]
    # Let's use a slightly tighter but safe range to help DE converge faster.
    log_low = -2.0
    log_high = 12.0
    bounds = [(log_low, log_high)] * 12

    best_val = 1e10
    best_log_roots = None
    best_roots = None

    # Initial guess: The geometric sequence from the seed
    seed_log_roots = np.log(4.0) + np.arange(12) * np.log(2.0)
    seed_log_roots = np.clip(seed_log_roots, log_low, log_high)
    
    # Evaluate the seed to have a baseline
    best_val = cost_func(seed_log_roots)
    best_log_roots = seed_log_roots.copy()
    best_roots = np.exp(seed_log_roots).copy()

    # Add roots of Laguerre polynomial L_12^(1) as another initial guess
    # L_n^(alpha)(x) = \sum_{k=0}^n (-1)^k \binom{n+alpha}{n-k} \frac{x^k}{k!}
    lag_coeffs_desc = np.zeros(13)
    for k in range(13):
        if k <= 12:
            lag_coeffs_desc[12 - k] = ((-1)**k) * math.comb(13, 12 - k) / math.factorial(k)
            
    try:
        lag_roots = np.roots(lag_coeffs_desc)
        lag_roots_real = lag_roots[np.abs(lag_roots.imag) < 1e-9].real
        lag_roots_pos = lag_roots_real[lag_roots_real > 0]
        if len(lag_roots_pos) == 12:
            lag_log_roots = np.log(lag_roots_pos)
            lag_log_roots = np.clip(lag_log_roots, log_low, log_high)
            val = cost_func(lag_log_roots)
            if val < best_val:
                best_val = val
                best_log_roots = lag_log_roots.copy()
                best_roots = np.exp(lag_log_roots).copy()
    except Exception:
        pass

    # Time allocation:
    # We will run DE for approximately 450 seconds.
    # Then polish for the remaining time.
    de_deadline = start_time + 450
    
    # Number of DE restarts
    num_de_runs = 4
    
    for run_idx in range(num_de_runs):
        if time.time() > de_deadline:
            break
            
        # Determine time limit for this specific DE run
        remaining_de_time = de_deadline - time.time()
        if run_idx == num_de_runs - 1:
            run_time_limit = remaining_de_time - 5
        else:
            run_time_limit = remaining_de_time / (num_de_runs - run_idx) - 5
            
        if run_time_limit < 10:
            continue

        # Use different seeds for each run
        seed = 42 + run_idx * 100
        
        def de_callback(xk, convergence):
            nonlocal best_val, best_log_roots, best_roots
            val = cost_func(xk)
            if val < best_val:
                best_val = val
                best_log_roots = xk.copy()
                best_roots = np.exp(xk)
            
            if time.time() > de_deadline:
                raise RuntimeError("Time limit reached")

        try:
            # 'sobol' initialization for better coverage
            de_result = differential_evolution(
                cost_func,
                bounds,
                seed=seed,
                maxiter=1000,
                popsize=15,
                tol=1e-8,
                polish=False,
                init='sobol',
                callback=de_callback,
                update_interval=1
            )
            
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_log_roots = de_result.x.copy()
                best_roots = np.exp(de_result.x)
                
        except RuntimeError:
            pass
        except Exception:
            pass

    # Final Polish Phase
    # Use the remaining time (up to deadline - 10s) for high-precision polishing
    if time.time() < deadline - 10 and best_log_roots is not None:
        try:
            polish_result = minimize(
                cost_func, 
                best_log_roots, 
                method='L-BFGS-B',
                bounds=bounds, 
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_log_roots = polish_result.x.copy()
                best_roots = np.exp(polish_result.x)
        except Exception:
            pass

    # If we still have time, try a few random restarts with L-BFGS-B
    if time.time() < deadline - 5:
        rng = np.random.default_rng(123)
        for _ in range(10):
            if time.time() > deadline - 2:
                break
            init = rng.uniform(log_low, log_high, size=12)
            try:
                res = minimize(
                    cost_func, init, method='L-BFGS-B',
                    bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_log_roots = res.x.copy()
                    best_roots = np.exp(res.x)
            except Exception:
                pass

    # Ensure roots are positive and sorted
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)
    
    # Absolute fallback
    return [4.0 * (2.0 ** i) for i in range(12)]
