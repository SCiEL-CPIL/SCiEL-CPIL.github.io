import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math

def construct():
    start_time = time.time()
    # The evaluator kills after 630s. We set a hard deadline at 600s to ensure
    # we have time to return the result and flush any output.
    deadline = start_time + 600

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
    # A is lower triangular, so A.T is upper triangular.
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
    # 1. Multiple Differential Evolution runs with different seeds to explore the landscape.
    # 2. Each DE run is time-bounded.
    # 3. After each DE run, polish the best candidate with L-BFGS-B.
    # 4. Keep track of the global best.

    # Define bounds for log_roots
    # Roots in [1e-3, 1e4] -> logs in [-6.9, 9.2]
    # Let's use a slightly tighter but safe range to help DE converge faster.
    # Based on the seed 4*2^i, logs are ~1.4 to ~9.
    # Let's allow [0.0, 10.0] for logs, i.e., roots [1, 22026].
    # Actually, let's be a bit wider to be safe: [ -2.0, 12.0 ]
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

    # Time allocation:
    # We will run DE for approximately 400 seconds, split among several runs.
    # Then polish for the remaining time.
    de_deadline = start_time + 420  # Leave 180s for polishing and safety
    
    # Number of DE restarts
    num_de_runs = 4
    
    for run_idx in range(num_de_runs):
        if time.time() > de_deadline:
            break
            
        # Determine time limit for this specific DE run
        remaining_de_time = de_deadline - time.time()
        # Each run gets a portion of the remaining time, but capped to prevent one run from hogging
        # If this is the last run, give it the rest (minus safety margin)
        if run_idx == num_de_runs - 1:
            run_time_limit = remaining_de_time - 5
        else:
            run_time_limit = remaining_de_time / (num_de_runs - run_idx) - 5
            
        if run_time_limit < 10:
            continue

        # Use different seeds for each run
        seed = 42 + run_idx * 100
        
        # Adjust popsize and maxiter based on available time?
        # For 12D, popsize=15 is reasonable. 
        # maxiter can be high, but we will cut it off with a callback.
        
        def de_callback(xk, convergence):
            nonlocal best_val, best_log_roots, best_roots
            # cost_func is called by DE internally, but we can check the current best
            # However, DE's callback receives the current best individual in the population.
            # We should evaluate it to be sure, or trust DE's internal tracking.
            # Let's evaluate to be safe and update our global best.
            val = cost_func(xk)
            if val < best_val:
                best_val = val
                best_log_roots = xk.copy()
                best_roots = np.exp(xk)
            
            # Check if we should stop this specific run early?
            # DE doesn't have a direct "stop" flag in callback, but we can raise an exception
            # to break out of the loop.
            if time.time() > de_deadline:
                raise RuntimeError("Time limit reached")

        try:
            # 'sobol' initialization for better coverage
            de_result = differential_evolution(
                cost_func,
                bounds,
                seed=seed,
                maxiter=1000, # High limit, will be cut by time
                popsize=15,
                tol=1e-8,
                polish=False, # We do our own polishing
                init='sobol',
                callback=de_callback,
                update_interval=1
            )
            
            # Update best from DE result if it's better
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_log_roots = de_result.x.copy()
                best_roots = np.exp(de_result.x)
                
        except RuntimeError:
            pass
        except Exception as e:
            # If DE fails, continue to next run
            pass

    # Final Polish Phase
    # Use the remaining time (up to deadline - 10s) for high-precision polishing
    if time.time() < deadline - 10 and best_log_roots is not None:
        try:
            # L-BFGS-B is good for smooth problems
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
    # This is a fallback if DE didn't find a good starting point for the final polish
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

    # Ensure roots are positive and sorted (order doesn't matter for product, but good for consistency)
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)
    
    # Absolute fallback
    return [4.0 * (2.0 ** i) for i in range(12)]
