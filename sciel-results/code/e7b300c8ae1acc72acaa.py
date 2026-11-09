"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling-inspired acquisition function.
    
    Instead of a fixed UCB or EI, this approach uses a dynamic exploration weight 
    that scales with the current best value. Early on, when f_best is high, the 
    weight is large to encourage exploration of high-variance regions. As f_best 
    approaches 0, the weight decreases to focus on exploitation.
    
    Additionally, we normalize sigma by its quantiles to ensure that the exploration 
    term is relative to the current uncertainty landscape, making the exploration 
    more robust to changes in the GP posterior.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    
    # Clip sigma to avoid division by zero or numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Dynamic exploration weight based on f_best
    # When f_best is large (early), beta is large. When f_best is small (late), beta is small.
    # We use a scaling that transitions from ~3.0 to ~0.5 as f_best goes from ~5 to ~0.
    # Ackley 2D in [-5, 5]^2 has max value around 20-30, but we focus on the transition to 0.
    # A simple linear or exponential decay works. Let's use a smooth transition.
    # beta = 0.5 + 2.5 * exp(-f_best / 0.1)
    # But we also want it to depend on the relative uncertainty.
    
    # Quantile-normalize sigma to make the exploration term adaptive to the current distribution
    # This ensures that we are exploring the "top" of the uncertainty distribution relative to the rest
    q_95 = np.percentile(sigma_safe, 95)
    q_50 = np.percentile(sigma_safe, 50)
    if q_95 > q_50:
        sigma_norm = (sigma_safe - q_50) / (q_95 - q_50)
    else:
        sigma_norm = sigma_safe
    
    # Ensure sigma_norm is non-negative and finite
    sigma_norm = np.maximum(sigma_norm, 0.0)
    sigma_norm = np.where(np.isfinite(sigma_norm), sigma_norm, 0.0)
    
    # Dynamic beta: starts high, decays as f_best improves
    # f_best starts around 5-10 and goes to 0.
    # Let's map f_best in [0, 10] to beta in [0.5, 3.0]
    if f_best > 10:
        beta = 3.0
    elif f_best < 0.1:
        beta = 0.5
    else:
        beta = 0.5 + 2.5 * (f_best / 10.0)
    
    # Thompson Sampling heuristic: -mu + beta * sigma_norm
    # We use the normalized sigma to ensure that the exploration is relative to the current uncertainty landscape
    score = -mu + beta * sigma_norm
    
    # Check time
    if time.time() > deadline:
        # If we are running out of time, return a simple exploitation score
        score = -mu
        score = np.where(np.isfinite(score), score, 0.0)
        return score
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
