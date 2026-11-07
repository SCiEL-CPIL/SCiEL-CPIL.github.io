"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    # Deadline guard: track elapsed time
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division issues
    sigma = np.maximum(sigma, 1e-9)
    
    # 1. Expected Improvement (EI)
    # z-score for EI
    z = (f_best - mu) / sigma
    # EI formula for minimization: (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Upper Confidence Bound (UCB) for minimization
    # Higher score means more likely to be selected.
    # For minimization, we want low mu. Uncertainty (sigma) is good for exploration.
    # UCB_min = -mu + beta * sigma
    beta = 2.0
    ucb = -mu + beta * sigma
    
    # 3. Normalize scores to zero mean and unit variance
    # This allows combining EI and UCB without scale bias
    ei_mean = np.mean(ei)
    ei_std = np.std(ei)
    if ei_std < 1e-9:
        ei_norm = np.zeros_like(ei)
    else:
        ei_norm = (ei - ei_mean) / ei_std
        
    ucb_mean = np.mean(ucb)
    ucb_std = np.std(ucb)
    if ucb_std < 1e-9:
        ucb_norm = np.zeros_like(ucb)
    else:
        ucb_norm = (ucb - ucb_mean) / ucb_std
        
    # 4. Combine
    # Equal weight initially. Could tune this.
    score = 0.5 * ei_norm + 0.5 * ucb_norm
    
    # Check deadline (safety margin 30s before 330s limit)
    # This computation is very fast, but we guard just in case.
    elapsed = time.time() - start_time
    if elapsed > 290:
        # Fallback to EI if somehow too slow (unlikely)
        return ei
        
    return score
# EVOLVE-BLOCK-END
