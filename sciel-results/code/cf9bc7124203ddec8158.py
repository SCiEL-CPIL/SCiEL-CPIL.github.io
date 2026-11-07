# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Hybrid acquisition: blend of EI and UCB for minimization
    # mu, sigma: arrays of shape (N,)
    # f_best: scalar, best value observed so far (lower is better)
    
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit
    
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB for minimization: lower mean + beta * uncertainty
    # We want to maximize the score, so we use -mu + beta * sigma
    # The beta parameter controls exploration vs exploitation
    # Tune beta to balance: too low = pure exploitation, too high = pure exploration
    beta = 1.5
    ucb = -mu + beta * sigma_safe
    
    # Normalize both to comparable scales
    # EI is already in objective units, UCB needs scaling
    # Use a weighted combination
    # Weight: 0.6 for EI (exploitation) + 0.4 for UCB (exploration)
    # This helps escape local minima while still converging to good regions
    
    # Scale EI and UCB to similar ranges for fair combination
    ei_min, ei_max = np.min(ei), np.max(ei)
    ucb_min, ucb_max = np.min(ucb), np.max(ucb)
    
    ei_range = ei_max - ei_min if ei_max > ei_min else 1.0
    ucb_range = ucb_max - ucb_min if ucb_max > ucb_min else 1.0
    
    # Normalize to [0, 1]
    ei_norm = (ei - ei_min) / ei_range
    ucb_norm = (ucb - ucb_min) / ucb_range
    
    # Combine: 60% exploitation (EI), 40% exploration (UCB)
    score = 0.6 * ei_norm + 0.4 * ucb_norm
    
    # Check deadline
    if time.time() > deadline:
        score = np.where(np.isfinite(score), score, 0.0)
        return score
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
