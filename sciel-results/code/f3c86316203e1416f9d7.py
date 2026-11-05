import numpy as np
import time
from scipy.optimize import minimize

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Initial guess: Multi-mode cosine
    # We want to minimize the peak of the autoconvolution.
    # A "flat" convolution is ideal.
    # Random initialization might get stuck in local minima.
    # Structured initialization is better.
    
    # Let's try a few different initial structures and pick the best one quickly?
    # Or just one good one.
    
    # h_i = sum of cosines
    # Frequencies: 1, 2, 3 cycles across N.
    i = np.arange(N)
    h0 = np.ones(N)
    h0 += 0.5 * np.cos(2 * np.pi * i / N)
    h0 += 0.2 * np.cos(4 * np.pi * i / N)
    h0 += 0.1 * np.cos(6 * np.pi * i / N)
    
    # Ensure sum is not 0. It's positive here.
    
    best_h = h0.copy()
    best_val = np.inf
    
    # Optimization loop
    # We will use a soft-max surrogate for the max.
    # Objective: minimize softmax_beta(conv(h_norm, h_norm))
    
    def objective(x):
        # x is the vector of heights
        s = np.sum(x)
        if abs(s) < 1e-10:
            return 1e10
        h_norm = x / s
        c = np.convolve(h_norm, h_norm, mode='full')
        # Softmax with beta
        # beta is captured from outer scope or passed?
        # In scipy minimize, we pass args.
        beta = args[0]
        
        c_max = np.max(c)
        # Numerical stability
        exp_c = np.exp(beta * (c - c_max))
        log_sum_exp = np.log(np.sum(exp_c)) + beta * c_max
        val = log_sum_exp / beta
        
        return val

    def objective_args(x, beta):
        return objective(x) # This doesn't work directly with minimize args if objective signature is fixed.
        # Better to define a closure or use a class.
        
    # Let's use a helper function
    def make_objective(beta):
        def obj(x):
            s = np.sum(x)
            if abs(s) < 1e-10:
                return 1e10
            h_norm = x / s
            c = np.convolve(h_norm, h_norm, mode='full')
            c_max = np.max(c)
            exp_c = np.exp(beta * (c - c_max))
            log_sum_exp = np.log(np.sum(exp_c)) + beta * c_max
            return log_sum_exp / beta
        return obj

    # We will iterate with increasing beta
    betas = [10, 50, 100, 200, 500, 1000]
    current_h = h0.copy()
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        obj_func = make_objective(beta)
        
        # Use L-BFGS-B
        # We don't have bounds, but we can add them to prevent huge values?
        # No, scale is invariant, so values can be anything.
        # However, numerical issues can arise if values are too large/small.
        # Normalizing inside obj handles this.
        
        try:
            res = minimize(obj_func, current_h, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-8})
            if res.success or res.fun < best_val:
                current_h = res.x
                # Calculate actual value
                s = np.sum(current_h)
                if abs(s) > 1e-10:
                    h_norm = current_h / s
                    c = np.convolve(h_norm, h_norm, mode='full')
                    val = np.max(c)
                    if val < best_val:
                        best_val = val
                        best_h = current_h.copy()
        except Exception:
            pass
            
    # Final check
    s = np.sum(best_h)
    if abs(s) < 1e-10:
        return np.ones(N) # Fallback
        
    # The task asks for h, not h_norm.
    # Value is invariant to scale.
    # We can return best_h.
    
    return best_h
