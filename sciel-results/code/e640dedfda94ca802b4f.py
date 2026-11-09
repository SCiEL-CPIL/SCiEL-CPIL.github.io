"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Deadline guard: if we are close to the limit, return a safe fallback.
    # We track a global start time to enforce the 300s limit with margin.
    if not hasattr(acquisition, '_start_time'):
        acquisition._start_time = time.time()
    
    elapsed = time.time() - acquisition._start_time
    # If we've been running for more than 295 seconds, return a safe fallback
    # to ensure we don't timeout.
    if elapsed > 295:
        # Fallback: use PI (Probability of Improvement) which is faster to compute
        # and still effective.
        z = (f_best - mu) / (sigma + 1e-9)
        return norm.cdf(z)
    
    # Use a hybrid acquisition function that balances exploration and exploitation.
    # We'll use a weighted combination of EI and PI, with a slight bias towards
    # exploitation (PI) as suggested by the knowledge that PI accelerates convergence.
    
    # Compute EI
    z_ei = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma * norm.pdf(z_ei)
    
    # Compute PI
    pi = norm.cdf(z_ei)
    
    # Normalize both to [0, 1] range for better combination
    # EI can be negative, so we shift and scale
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    if ei_max - ei_min > 1e-12:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros_like(ei)
    
    # PI is already in [0, 1]
    
    # Combine with a weight favoring PI (exploitation)
    # As we progress, we could increase PI weight, but for simplicity, use a fixed weight
    weight_pi = 0.7
    weight_ei = 1.0 - weight_pi
    
    scores = weight_pi * pi + weight_ei * ei_norm
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
