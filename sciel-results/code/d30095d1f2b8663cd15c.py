"""Thompson Sampling acquisition for cost-aware Bayesian optimization."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling: draw one sample from GP posterior, select candidate
    minimizing sample / cost.
    
    Parameters:
        mu: (2000,) GP posterior mean in objective units
        sigma: (2000,) GP posterior std in objective units
        f_best: float, lowest observed objective value
        cost: (2000,) evaluation cost for each candidate
    
    Returns:
        (2000,) scores, higher = evaluate next
    """
    # Safety: ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and negative values
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Draw a single sample from the GP posterior: y = mu - sigma * z
    # (minimization: lower is better, so we sample and want to minimize)
    z = np.random.standard_normal(mu.shape)
    sample = mu - sigma * z
    
    # Thompson sampling score: minimize sample / cost
    # We return negative of (sample / cost) so higher score = better
    # But we need to handle cases where sample could be very negative (good)
    ts_score = -sample / cost
    
    # Ensure all outputs are finite
    ts_score = np.where(np.isfinite(ts_score), ts_score, -1e10)
    
    return ts_score
# EVOLVE-BLOCK-END
