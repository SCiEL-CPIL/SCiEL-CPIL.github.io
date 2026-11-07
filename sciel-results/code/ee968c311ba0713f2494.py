"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: ensure we don't hang, though vectorized ops are fast.
    # The total budget is 330s. This function is called ~25 times.
    # We allow 5 seconds per call as a hard safety limit.
    start_time = time.time()
    max_time = 5.0
    
    # Ensure sigma is positive to avoid division issues or log issues if used
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # UCB-based acquisition: Balance exploitation (-mu) and exploration (beta * sigma)
    # Since we want to minimize, we maximize -mu.
    # Cost-adjusted: divide by cost to prefer cheaper evaluations.
    
    # Adaptive beta: Start with high exploration, but we don't have step count.
    # We can use the variance of mu to gauge convergence.
    # If mu is very spread out, we are in early stages -> more exploration.
    mu_std = np.std(mu)
    # Scale beta based on the magnitude of the function values
    # If f_best is large, sigma might be large.
    # Let's use a fixed beta for stability, tuned for small budgets.
    beta = 2.5
    
    # Calculate UCB score
    ucb_score = -mu + beta * sigma
    
    # Cost adjustment
    score = ucb_score / cost
    
    # Safety check: if computation took too long (shouldn't happen), return a fallback
    if time.time() - start_time > max_time:
        # Fallback: pure exploitation
        score = -mu / cost
        
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
