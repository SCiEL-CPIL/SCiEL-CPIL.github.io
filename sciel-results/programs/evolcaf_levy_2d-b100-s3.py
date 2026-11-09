"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) with adaptive exploration.
    
    Score = (beta * sigma - mu) / cost
    Higher score is better.
    beta adapts to f_best: high exploration when f_best is high (early),
    low exploration when f_best is low (late).
    """
    # Safety guard for time (though this function is fast, it's good practice)
    t0 = time.time()
    
    # Flatten inputs
    mu = np.asarray(mu).flatten()
    sigma = np.asarray(sigma).flatten()
    cost = np.asarray(cost).flatten()
    
    # Safe division
    cost_safe = np.maximum(cost, 1e-9)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Adaptive exploration coefficient
    # f_best is the current minimum found.
    # Early in optimization, f_best is large. We want to explore (high beta).
    # Late in optimization, f_best is small. We want to exploit (low beta).
    
    # Use an exponential saturation function.
    # tau controls the speed of transition.
    # beta_min is the floor for exploitation.
    # beta_max is the ceiling for exploration.
    
    tau = 5.0
    beta_min = 0.1
    beta_max = 2.0
    
    # 1 - exp(-f_best / tau) goes from 0 (f_best=0) to 1 (f_best->inf)
    # So beta = beta_min + (beta_max - beta_min) * (1 - exp(-f_best / tau))
    # If f_best is large, beta ~ beta_max.
    # If f_best is small, beta ~ beta_min.
    
    # Handle potential overflow or edge cases in exp
    f_best_val = float(f_best)
    if f_best_val < 0:
        f_best_val = 0.0
        
    # Clip f_best for stability in exp, though exp(-large) is just 0
    # If f_best is huge, exp(-f_best/tau) is 0, term is 1.
    exponent = -f_best_val / tau
    if exponent < -50:
        sat = 1.0
    else:
        sat = 1.0 - np.exp(exponent)
        
    beta = beta_min + (beta_max - beta_min) * sat
    
    # LCB for minimization: mu - beta * sigma
    # We want to MAXIMIZE the acquisition score.
    # So we negate the LCB: -(mu - beta * sigma) = beta * sigma - mu
    lcb_score = beta * sigma_safe - mu
    
    # Cost-aware normalization
    score = lcb_score / cost_safe
    
    # Ensure finiteness
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Time check (unlikely to be needed for this vectorized op)
    elapsed = time.time() - t0
    if elapsed > 250:
        return score
        
    return score
# EVOLVE-BLOCK-END
