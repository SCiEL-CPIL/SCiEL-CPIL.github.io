# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling acquisition with cost-awareness.
    
    Draws a posterior sample for each candidate and scores by
    (sample - f_best) / cost, favoring promising low-cost points.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Draw a posterior sample: x_t ~ N(mu, sigma^2)
    # Use a fixed seed based on the current state for reproducibility within a run
    # But for Thompson sampling, we want randomness to escape local optima
    # Use the current time as a seed source for variety
    rng = np.random.default_rng()
    samples = mu + sigma_safe * rng.standard_normal(len(mu))
    
    # Score: how much better than f_best is this sample, per unit cost
    # Lower f_best is better, so (f_best - sample) is positive when sample < f_best
    # Dividing by cost gives expected improvement per unit cost
    score = (f_best - samples) / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
