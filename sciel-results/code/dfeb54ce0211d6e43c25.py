import numpy as np
import scipy.optimize
import time

N = 100

def construct():
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer for safety
    
    # Pre-compute indices for correlation if needed, but np.correlate is fast for N=100
    # We want to minimize max_k C_k where C_k = sum_i h_i (1 - h_{i+k})
    # Note: The problem defines overlap with shifted 1-h.
    # C(k) = sum_{i} h[i] * (1 - h[(i+k) % N]) if circular.
    # The seed program used linear correlation: np.correlate(h, 1-h, 'full').
    # Let's stick to the linear definition used in the seed to be safe, 
    # as "shifted" usually implies linear shifts in this context unless "circular" is specified.
    # However, the seed's proof for 0.5 used circular logic (sum over all shifts).
    # Let's check the seed's code:
    # "c = np.correlate(h, g, mode='full')"
    # "return max(c[:N-1], c[N:])"
    # This is linear correlation.
    
    def get_correlations(h):
        """
        Compute linear correlations of h and 1-h for all non-zero shifts.
        Returns an array of correlation values for shifts k = -(N-1) ... -1, 1 ... N-1.
        """
        g = 1.0 - h
        # mode='full' gives 2N-1 outputs.
        # Index N-1 is shift 0.
        # Indices 0 to N-2 are negative shifts.
        # Indices N to 2N-2 are positive shifts.
        c = np.correlate(h, g, mode='full')
        # Exclude shift 0
        c_nonzero = np.concatenate([c[:N-1], c[N:]])
        return c_nonzero

    def objective_and_grad(h, beta, lambda_sum):
        """
        Objective: LSE(correlations) + penalty(sum - N/2)^2
        Returns value and gradient (None for numerical).
        """
        c_all = get_correlations(h)
        
        # LogSumExp for numerical stability
        max_c = np.max(c_all)
        # Prevent overflow in exp
        if max_c > 500:
             max_c = 500
        # lse = max_c + (1/beta) * log(sum(exp(beta * (c_k - max_c))))
        # We use beta as the temperature parameter in the denominator
        lse = max_c + np.log(np.sum(np.exp(beta * (c_all - max_c)))) / beta
        
        # Sum penalty
        sum_h = np.sum(h)
        sum_pen = lambda_sum * (sum_h - N/2)**2
        
        val = lse + sum_pen
        
        # We will use numerical gradient to ensure correctness and simplicity,
        # as N=100 is small enough for L-BFGS-B to handle it within time limits
        # if we limit iterations.
        return val, None

    def optimize_instance(h0, beta, lambda_sum, max_iter):
        bounds = [(0, 1) for _ in range(N)]
        
        # Define objective function closure
        def obj(h):
            val, _ = objective_and_grad(h, beta, lambda_sum)
            return val

        try:
            res = scipy.optimize.minimize(
                obj, 
                h0, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            h_opt = res.x
        except Exception:
            h_opt = h0
        
        # Post-processing: Ensure sum is exactly N/2 and bounds are respected
        h_opt = np.clip(h_opt, 0, 1)
        
        # Correct sum
        # We want sum(h) = N/2
        # Add a constant to all elements, then clip and iterate
        for _ in range(10):
            current_sum = np.sum(h_opt)
            diff = N/2 - current_sum
            if abs(diff) < 1e-12:
                break
            h_opt += diff / N
            h_opt = np.clip(h_opt, 0, 1)
        
        return h_opt

    def score_h(h):
        """Calculate the true max overlap score."""
        c_all = get_correlations(h)
        return np.max(c_all)

    best_h = np.full(N, 0.5)
    best_score = score_h(best_h) # N/4 = 25.0

    # Multi-restart strategy
    # Start with uniform, then perturb
    rng = np.random.default_rng(42)
    
    # List of restart configurations
    # (initial_h, beta, lambda_sum, max_iter)
    restarts = []
    
    # 1. Uniform start (baseline, might stay at saddle)
    restarts.append((np.full(N, 0.5), 5.0, 100.0, 500))
    
    # 2-5. Random perturbations to escape saddle
    for i in range(4):
        # Generate random h in [0.1, 0.9] to avoid boundaries initially
        h_rand = rng.uniform(0.1, 0.9, size=N)
        # Normalize to sum N/2 roughly
        h_rand = h_rand / np.sum(h_rand) * (N/2)
        h_rand = np.clip(h_rand, 0, 1)
        
        # Increase beta to sharpen the max
        beta = 5.0 + i * 5.0
        # Increase lambda_sum to enforce constraint more strictly
        lambda_sum = 100.0 + i * 50.0
        
        restarts.append((h_rand, beta, lambda_sum, 500))
        
    # 6-10. More aggressive perturbations and higher beta
    for i in range(5):
        # Binary-like initializations
        h_bin = rng.choice([0.0, 1.0], size=N)
        # Adjust to sum N/2
        # Count ones
        ones = np.sum(h_bin == 1)
        # We need N/2 ones. If we have more, flip some 1s to 0s. If less, flip 0s to 1s.
        # This is a heuristic. Let's just use random uniform but sharper.
        h_rand = rng.uniform(0.0, 1.0, size=N)
        # Make it more bimodal
        h_rand = np.where(h_rand > 0.5, 0.9, 0.1)
        h_rand = h_rand / np.sum(h_rand) * (N/2)
        h_rand = np.clip(h_rand, 0, 1)
        
        beta = 15.0 + i * 5.0
        lambda_sum = 200.0 + i * 50.0
        
        restarts.append((h_rand, beta, lambda_sum, 500))

    for idx, (h0, beta, lambda_sum, max_iter) in enumerate(restarts):
        if time.time() - start_time > time_limit:
            break
            
        h_opt = optimize_instance(h0, beta, lambda_sum, max_iter)
        
        # Calculate true score
        current_score = score_h(h_opt)
        
        # Verify constraint
        if not np.isclose(np.sum(h_opt), N/2, atol=1e-6):
            continue
        if not (np.all(h_opt >= 0) and np.all(h_opt <= 1)):
            continue
            
        if current_score < best_score:
            best_score = current_score
            best_h = h_opt.copy()
            
    return best_h
