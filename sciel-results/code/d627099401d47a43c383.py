import numpy as np
from scipy.optimize import minimize
import time

N = 100
TARGET_SUM = 50.0

def _correlation_value(h):
    """Compute the value as defined in the task."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _smooth_max_objective(h, beta, lambda_sum):
    """
    Smooth approximation of the max correlation + penalty for sum constraint.
    h: array of shape (N,)
    """
    # Compute all correlations
    corr = np.correlate(h, 1 - h, mode="full")
    
    # Softmax approximation of max
    # To prevent overflow, subtract max
    c_max = np.max(corr)
    if c_max > 1e6:
        return 1e9
    exp_corr = np.exp(beta * (corr - c_max))
    log_sum_exp = c_max + np.log(np.sum(exp_corr)) / beta
    
    # Penalty for sum constraint
    sum_err = np.sum(h) - TARGET_SUM
    penalty = lambda_sum * sum_err**2
    
    return log_sum_exp + penalty

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)
    
    # List of beta values to try (increasing sharpness)
    betas = [10, 50, 100, 200, 500]
    lambda_sum = 100.0
    
    # Random restarts
    # We will try a few different initializations
    initializations = [
        np.full(N, 0.5),
        np.random.rand(N) * 0.2 + 0.4, # Near 0.5
        np.concatenate([np.ones(N//2), np.zeros(N//2)]), # Block
        np.sin(np.linspace(0, 4*np.pi, N)) * 0.25 + 0.5, # Sine
        np.sin(np.linspace(0, 8*np.pi, N)) * 0.25 + 0.5  # Sine 2x
    ]
    
    # Ensure initializations are within [0,1] and close to sum 50
    # We'll let the optimizer fix the sum, but good starts help.
    for init in initializations:
        if time.time() > deadline:
            break
            
        h0 = np.clip(init, 0, 1)
        # Normalize to sum 50 roughly?
        # No, let the penalty handle it.
        
        for beta in betas:
            if time.time() > deadline:
                break
                
            # Optimize
            # Use L-BFGS-B
            res = minimize(
                _smooth_max_objective, 
                h0, 
                args=(beta, lambda_sum),
                method='L-BFGS-B', 
                bounds=[(0, 1)] * N,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            h_opt = res.x
            # Project to feasible set: clip to [0,1] and adjust sum
            h_opt = np.clip(h_opt, 0, 1)
            
            # Adjust sum to be exactly 50
            # Simple iterative adjustment
            for _ in range(10):
                diff = TARGET_SUM - np.sum(h_opt)
                if abs(diff) < 1e-9:
                    break
                # Add diff/N to all, then clip
                h_opt += diff / N
                h_opt = np.clip(h_opt, 0, 1)
            
            val = _correlation_value(h_opt)
            if val < best_val:
                best_val = val
                best_h = h_opt.copy()
        
        # Update h0 for next beta/restart? 
        # Keep the best found so far as the new start for higher beta
        h0 = best_h.copy()
        
    # Final refinement with high beta
    if time.time() < deadline - 10:
        beta_final = 1000
        res = minimize(
            _smooth_max_objective, 
            best_h, 
            args=(beta_final, lambda_sum),
            method='L-BFGS-B', 
            bounds=[(0, 1)] * N,
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-10}
        )
        h_final = np.clip(res.x, 0, 1)
        # Adjust sum
        for _ in range(10):
            diff = TARGET_SUM - np.sum(h_final)
            if abs(diff) < 1e-9:
                break
            h_final += diff / N
            h_final = np.clip(h_final, 0, 1)
        
        val_final = _correlation_value(h_final)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final.copy()

    return best_h
