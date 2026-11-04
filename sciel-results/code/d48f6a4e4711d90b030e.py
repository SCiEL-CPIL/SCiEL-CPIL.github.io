"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s safety margin within 150s limit
    
    # Phase 1: Continuous Optimization
    # Minimize sum of squared non-zero autocorrelation lags
    # This promotes spectral flatness, which reduces the max autocorrelation peak
    
    def objective(x):
        c = np.convolve(x, x, mode='full')
        # c[N-1] is lag 0
        # We want to minimize sum(c^2) - c[N-1]^2
        return np.sum(c*c) - c[N-1]*c[N-1]
        
    # Initial guess: Random signs
    rng = np.random.RandomState(42)
    x0 = rng.choice([-1.0, 1.0], size=N)
    # Normalize to sum=1 to help optimization
    x0 = x0 / np.sum(x0)
    
    x_opt = x0.copy()
    initial_obj = objective(x0)
    
    # Use L-BFGS-B
    # Limit time for this phase
    phase1_deadline = start_time + 30.0
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', 
                       options={'maxiter': 100, 'maxfun': 500, 'disp': False})
        if res.fun < initial_obj:
            x_opt = res.x
    except Exception:
        pass
        
    # Project to signs
    x = np.sign(x_opt)
    # Handle zeros
    x[x == 0] = 1.0
    
    # Ensure sum is not zero
    if abs(np.sum(x)) < 1e-10:
        x[0] = -x[0] # Flip one to break symmetry

    # Phase 2: Local Search
    s = np.sum(x)
    c = np.convolve(x, x, mode='full')
    max_abs_c = np.max(np.abs(c))
    
    best_score = max_abs_c / (s * s)
    best_x = x.copy()
    best_c = c.copy()
    best_s = s
    best_max_abs = max_abs_c
    
    improved = True
    pass_count = 0
    while improved and (time.time() - start_time) < deadline - 5.0:
        improved = False
        pass_count += 1
        
        # Randomize order to avoid systematic bias
        indices = rng.permutation(N)
        
        for i in indices:
            if (time.time() - start_time) > deadline - 5.0:
                break
            
            old_val = x[i]
            new_val = -old_val
            delta = new_val - old_val  # -2*old_val
            
            # Update sum
            new_s = s + delta
            
            # Recompute autocorrelation
            x[i] = new_val
            new_c = np.convolve(x, x, mode='full')
            new_max_abs = np.max(np.abs(new_c))
            
            if abs(new_s) > 1e-10:
                new_score = new_max_abs / (new_s * new_s)
            else:
                new_score = float('inf')
                
            if new_score < best_score:
                best_score = new_score
                best_x = x.copy()
                best_c = new_c.copy()
                best_s = new_s
                best_max_abs = new_max_abs
                s = new_s
                c = new_c
                max_abs_c = new_max_abs
                improved = True
            else:
                x[i] = old_val
                # c and max_abs_c remain as they were (best state)
                # x is reverted, so it matches best_x state for this index
                
    return best_x
# EVOLVE-BLOCK-END
