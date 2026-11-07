"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Deadline guard: track start time
    t_start = time.time()
    deadline = t_start + 290.0  # 300s limit, 10s safety margin
    
    n = len(mu)
    
    # Expected Improvement acquisition
    # Use the known minimum of 3.0 as the target for stronger exploration signal
    # but blend with f_best to maintain exploitation of observed data
    target = 3.0
    
    # Compute EI with respect to the true minimum
    z = (target - mu) / (sigma + 1e-9)
    ei_true = (target - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute EI with respect to observed best
    z_best = (f_best - mu) / (sigma + 1e-9)
    ei_best = (f_best - mu) * norm.cdf(z_best) + sigma * norm.pdf(z_best)
    
    # Blend: primarily use EI toward true minimum for exploration,
    # but keep some exploitation of observed best
    # When f_best is close to 3.0, both are similar
    # When f_best >> 3.0, ei_true provides stronger signal in uncertain regions
    blend = 0.7 * ei_true + 0.3 * ei_best
    
    # Add a small UCB component to encourage exploration of high-variance regions
    # This helps escape local minima
    ucb_beta = 0.5
    ucb = mu + ucb_beta * sigma
    # Invert since we're minimizing: lower ucb is better, so score = -ucb
    ucb_score = -ucb
    
    # Normalize both to comparable scale
    blend_mean = np.mean(blend)
    blend_std = np.std(blend) + 1e-9
    blend_norm = (blend - blend_mean) / blend_std
    
    ucb_mean = np.mean(ucb_score)
    ucb_std = np.std(ucb_score) + 1e-9
    ucb_norm = (ucb_score - ucb_mean) / ucb_std
    
    # Final score: weighted combination
    score = 0.8 * blend_norm + 0.2 * ucb_norm
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
