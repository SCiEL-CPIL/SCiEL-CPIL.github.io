"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Constructs a sequence of step heights h of length N such that:
    1. 0 <= h_i <= 1
    2. sum(h) == N/2
    3. Minimizes the maximum overlap of h with shifted 1-h.
    
    Overlap for shift k is defined as sum_i h_i * (1 - h_{i+k}) using linear correlation.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Precompute bounds for optimizer
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    best_h = np.full(N, 0.5)
    best_score = float('inf')
    
    def get_correlations(h):
        """Compute linear correlations of h and 1-h for all non-zero shifts."""
        g = 1.0 - h
        c = np.correlate(h, g, mode='full')
        # Exclude center (shift 0)
        c1 = c[:N-1]
        c2 = c[N:]
        return np.concatenate([c1, c2])

    def true_score(h):
        """Calculate the raw score (max overlap)."""
        c_all = get_correlations(h)
        return np.max(c_all)

    def project_to_feasible(h):
        """Project h to satisfy sum(h) = N/2 and 0 <= h <= 1."""
        h = np.clip(h, 0.0, 1.0)
        target_sum = N / 2.0
        current_sum = np.sum(h)
        diff = target_sum - current_sum
        
        if diff != 0:
            # Simple iterative adjustment
            for _ in range(10):
                if abs(diff) < 1e-12:
                    break
                h += diff / N
                h = np.clip(h, 0.0, 1.0)
                current_sum = np.sum(h)
                diff = target_sum - current_sum
        return h

    # Initial evaluation
    best_score = true_score(best_h)
    
    # Optimization strategy:
    # Use scipy.optimize.minimize with L-BFGS-B.
    # Objective: Smooth approximation of max overlap (LogSumExp) + penalty for sum constraint.
    # We vary the "temperature" beta to sharpen the max approximation.
    
    import scipy.optimize
    
    def make_objective(beta, lambda_sum):
        def obj(h):
            c_all = get_correlations(h)
            max_c = np.max(c_all)
            # LogSumExp for numerical stability
            # lse = max_c + log(sum(exp(c - max_c))) / beta
            if beta > 0:
                exp_c = np.exp(beta * (c_all - max_c))
                lse = max_c + np.log(np.sum(exp_c)) / beta
            else:
                lse = max_c
            
            sum_h = np.sum(h)
            sum_pen = lambda_sum * (sum_h - N/2.0)**2
            
            return lse + sum_pen
        return obj

    # List of configurations to try
    # (beta, lambda_sum, initial_h, description)
    configs = [
        (5.0, 100.0, np.full(N, 0.5), "uniform_start_low_beta"),
        (10.0, 100.0, np.full(N, 0.5), "uniform_start_mid_beta"),
        (20.0, 100.0, np.full(N, 0.5), "uniform_start_high_beta"),
        (50.0, 100.0, np.full(N, 0.5), "uniform_start_vhigh_beta"),
    ]
    
    # Add randomized restarts
    rng = np.random.default_rng(42)
    for i in range(20):
        # Generate random h in [0.2, 0.8] to stay away from boundaries initially
        h_rand = rng.uniform(0.2, 0.8, N)
        h_rand = project_to_feasible(h_rand)
        beta = 10.0 + i * 5.0
        configs.append((beta, 100.0, h_rand, f"rand_restart_{i}"))
    
    # Add specific structured patterns that might help
    # Sine wave
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    h_sine = 0.5 + 0.4 * np.sin(t)
    h_sine = project_to_feasible(h_sine)
    configs.append((10.0, 100.0, h_sine, "sine_pattern"))
    
    # Sawtooth
    h_saw = np.linspace(0.1, 0.9, N)
    h_saw = project_to_feasible(h_saw)
    configs.append((10.0, 100.0, h_saw, "sawtooth_pattern"))

    for beta, lam, h_init, desc in configs:
        if time.time() > deadline:
            break
        
        h_init = np.array(h_init, dtype=float)
        h_init = project_to_feasible(h_init)
        
        obj_func = make_objective(beta, lam)
        
        try:
            res = scipy.optimize.minimize(
                obj_func, 
                h_init, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            h_opt = res.x
            h_opt = project_to_feasible(h_opt)
            
            # Evaluate true score
            score = true_score(h_opt)
            
            if score < best_score:
                best_score = score
                best_h = h_opt
                # print(f"New best: {best_score:.6f} from {desc}")
                
        except Exception as e:
            pass # Ignore optimization failures
    
    # Final refinement: try to do a local polish on the best found so far
    if time.time() < deadline - 5.0:
        # Use a very high beta to sharpen the max
        high_beta = 100.0
        obj_func = make_objective(high_beta, 1000.0)
        
        try:
            res = scipy.optimize.minimize(
                obj_func, 
                best_h, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_opt = res.x
            h_opt = project_to_feasible(h_opt)
            score = true_score(h_opt)
            if score < best_score:
                best_score = score
                best_h = h_opt
        except:
            pass

    return best_h
# EVOLVE-BLOCK-END
