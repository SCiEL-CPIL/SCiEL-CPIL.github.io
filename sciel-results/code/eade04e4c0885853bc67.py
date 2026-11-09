"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6d (minimization).
    
    Uses a hybrid strategy:
    - Expected Improvement (EI) for exploitation
    - UCB-like term (kappa * sigma) for exploration
    - Dynamic kappa that adapts based on the spread of the GP predictions
    
    mu: (n,) array of GP mean predictions
    sigma: (n,) array of GP standard deviation predictions
    f_best: current best observed value (minimum)
    
    Returns: (n,) array of scores, higher is better
    """
    start_time = time.time()
    deadline = start_time + 295.0  # 5s safety margin before 300s limit

    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)

    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    EI = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    EI = np.maximum(EI, 0.0)

    # Dynamic exploration coefficient based on GP uncertainty spread
    # Early on, sigma values are larger and more spread out -> explore more
    # Later, sigma values are smaller and concentrated -> exploit more
    
    # Measure how much uncertainty remains relative to the best found value
    # The objective range for Hartmann 6d is roughly [-3.32, 0.5], so scale ~3.8
    obj_scale = 3.8
    
    # Use the median sigma as a robust measure of current uncertainty level
    sigma_median = np.median(sigma_safe)
    
    # When sigma_median is large (early), kappa is large (explore)
    # When sigma_median is small (late), kappa is small (exploit)
    # Map sigma_median from [0, obj_scale] to kappa in [0.5, 2.5]
    sigma_ratio = np.clip(sigma_median / obj_scale, 0.0, 1.0)
    kappa = 0.5 + 2.0 * sigma_ratio
    
    # UCB term for exploration
    UCB = kappa * sigma_safe
    
    # Combined score: EI + UCB
    # EI captures the expected improvement, UCB captures uncertainty
    scores = EI + UCB

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        return scores

    return scores
# EVOLVE-BLOCK-END
