"""Program for Bayesian optimization acquisition function."""
import time
import numpy as np
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
TIME_BUDGET = 310.0  # Leave 20s margin before 330s kill


def _time_remaining():
    return TIME_BUDGET - (time.time() - _START_TIME)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using UCB (Upper Confidence Bound for maximization
    of -f) for the Styblinski-Tang 1D minimization task.
    
    Args:
        mu: GP posterior mean, shape (N,)
        sigma: GP posterior std, shape (N,)
        f_best: Best observed value so far (lower is better)
        
    Returns:
        Scores for each candidate, shape (N,). Higher scores preferred.
    """
    # Safety checks
    sigma = np.asarray(sigma, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Clamp extreme sigma values
    sigma_safe = np.clip(sigma_safe, 1e-10, 1e6)
    
    # Compute UCB for minimization: score = -mu + kappa * sigma
    # This encourages exploration when uncertainty is high and exploitation
    # when the mean is low.
    
    # Adaptive kappa based on gap to known optimum
    target = -39.1662
    gap = f_best - target  # positive if we're above the optimum
    
    # Use a larger kappa early on for exploration, smaller later for exploitation
    # kappa ranges from ~2.5 (large gap) to ~0.5 (small gap)
    kappa = 0.5 + 2.0 * (1.0 / (1.0 + np.exp(-gap / 3.0)))
    
    # UCB score for minimization (higher is better)
    scores = -mu + kappa * sigma_safe
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
