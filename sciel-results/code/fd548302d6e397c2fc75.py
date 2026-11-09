import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d on [-5, 5].
    
    Uses Posterior Mean Minimization (PMM) as the sole strategy.
    This ignores the predictive standard deviation (sigma) and focuses
    entirely on minimizing the predictive mean (mu). This approach is
    highly effective for smooth, low-dimensional functions where the
    GP's mean provides a strong gradient towards the minimum.
    """
    mu = np.asarray(mu, dtype=np.float64)
    
    # Ensure mu contains only finite values
    # Replace NaNs with 0.0, +inf with a large positive number, -inf with a large negative number
    # Since we return -mu, we want the lowest mu to have the highest score.
    # If mu is +inf, -mu is -inf (bad).
    # If mu is -inf, -mu is +inf (good).
    # However, mu should be finite.
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Score: Higher is better. We want to minimize mu, so we return -mu.
    scores = -mu
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
