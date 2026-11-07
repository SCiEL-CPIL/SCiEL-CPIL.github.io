"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_deadline = _start_time + 290.0

def acquisition(mu, sigma, f_best):
    # Deadline guard: if we are running out of time, fall back to greedy (minimize mu)
    if time.time() > _deadline:
        return -mu
    
    # Ensure sigma is strictly non-negative and finite
    sigma = np.maximum(sigma, 0.0)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure mu is finite
    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Adaptive UCB for minimization
    # We want to minimize f(x).
    # Uncertainty is sigma.
    # Optimistic model: assume the true value is mu - beta * sigma.
    # Equivalently, maximize: -mu(x) + beta * sigma(x)
    
    # Adaptive beta:
    # - Early on (high f_best), we want more exploration (higher beta).
    # - Later on (low f_best), we want more exploitation (lower beta).
    # Ackley 2D has a global minimum of 0.
    # If f_best is close to 0, we are likely near the global minimum, so exploit.
    # If f_best is high, we are far, so explore.
    
    # Normalize f_best relative to a scale.
    # Ackley 2D range is roughly [0, ~20].
    # Let's use a soft transition.
    # beta = beta_min + (beta_max - beta_min) * exp(-f_best / scale)
    
    # When f_best is large, exp term is ~0, beta ~ beta_min.
    # Wait, that's backwards.
    # If f_best is large (far from 0), we want HIGH beta (explore).
    # If f_best is small (close to 0), we want LOW beta (exploit).
    
    # So: beta = beta_min + (beta_max - beta_min) * (1 - exp(-f_best / scale))
    # If f_best -> 0, exp(0)=1, beta -> beta_min.
    # If f_best -> inf, exp(-inf)=0, beta -> beta_max.
    
    beta_min = 0.5
    beta_max = 2.0
    scale = 1.0
    
    # f_best can be negative in some contexts, but Ackley is >= 0.
    # Clamp f_best to non-negative for the exponential.
    f_best_pos = max(0.0, f_best)
    
    beta = beta_min + (beta_max - beta_min) * (1.0 - np.exp(-f_best_pos / scale))
    
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
