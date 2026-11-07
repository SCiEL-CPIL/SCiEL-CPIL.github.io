"""Rewritten acquisition function using a different approach family (Upper Confidence Bound with adaptive exploration)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6D.
    
    Uses an Upper Confidence Bound (UCB) style acquisition with an adaptive
    exploration parameter that balances exploitation and exploration based on
    the current best value and uncertainty levels.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at candidate points (in objective units, lower is better).
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Lowest observed objective value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising to evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure finite values
    sigma = np.maximum(sigma, 1e-9)
    
    # UCB acquisition: we want to minimize, so we use a lower confidence bound
    # LCB = mu - kappa * sigma
    # But we can also frame it as maximizing (f_best - mu) + kappa * sigma
    # which is equivalent to minimizing mu - kappa * sigma
    
    # Adaptive kappa based on the gap between f_best and the mean predictions
    # When f_best is much better than most predictions, we exploit more (smaller kappa)
    # When uncertainty is high relative to the gap, we explore more (larger kappa)
    
    # Compute a baseline gap
    gap = f_best - mu  # positive when prediction is worse than best
    
    # Estimate typical uncertainty relative to the gap
    median_sigma = np.median(sigma)
    median_gap = np.median(np.abs(gap))
    
    # Adaptive kappa: start from a reasonable base and adjust
    # Base kappa around 1.5-2.5 is common; we adjust based on uncertainty
    if median_gap > 1e-8:
        uncertainty_ratio = median_sigma / median_gap
        # When uncertainty is high relative to gap, explore more
        kappa = 1.0 + 1.5 * min(uncertainty_ratio, 2.0)
    else:
        # If gap is very small, be more exploratory
        kappa = 2.5
    
    # Clip kappa to a reasonable range
    kappa = np.clip(kappa, 0.5, 4.0)
    
    # LCB acquisition: minimize mu - kappa * sigma
    # Equivalent to maximizing -mu + kappa * sigma
    # Or equivalently, we can use: (f_best - mu) + kappa * sigma
    # which gives higher scores to points with lower mu and higher sigma
    
    # Use the form that naturally incorporates f_best for scaling
    scores = (f_best - mu) + kappa * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
