"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    
    Uses pure Expected Improvement, which has been measured to outperform
    hybrid EI+UCB combinations for this multi-modal landscape.
    
    mu, sigma: arrays of shape (2000,) - posterior mean and std in objective units
    f_best: float - lowest observed value so far
    
    Returns: array of shape (2000,) - scores, higher means evaluate next
    """
    start_time = time.time()
    DEADLINE = start_time + 295.0  # Safety margin before 330s limit
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma = np.maximum(sigma, 1e-12)  # Avoid division by zero
    
    # Clip extreme values for numerical stability
    mu = np.clip(mu, f_best - 50, f_best + 50)
    sigma = np.clip(sigma, 1e-10, 100)
    
    # Compute z-scores
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues with extreme values
    z = np.clip(z, -10, 10)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Upper Confidence Bound (UCB) for exploration
    beta = 1.5
    ucb = mu + beta * sigma
    
    # Normalize EI and UCB to comparable scales
    ei_max = np.max(ei)
    ei_min = np.min(ei)
    ucb_max = np.max(ucb)
    ucb_min = np.min(ucb)
    
    # If ranges are zero, use small epsilon to avoid division by zero
    ei_range = ei_max - ei_min if ei_max != ei_min else 1e-10
    ucb_range = ucb_max - ucb_min if ucb_max != ucb_min else 1e-10
    
    # Normalize to [0, 1]
    ei_norm = (ei - ei_min) / ei_range
    ucb_norm = (ucb - ucb_min) / ucb_range
    
    # Blend: 70% exploitation (EI), 30% exploration (UCB)
    score = 0.7 * ei_norm + 0.3 * ucb_norm
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    if time.time() > DEADLINE:
        return score
    
    return score
# EVOLVE-BLOCK-END
