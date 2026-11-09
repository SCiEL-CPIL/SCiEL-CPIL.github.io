# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Upper Confidence Bound (UCB) for minimization.
    
    We transform the minimization problem into a maximization problem by
    negating the mean. The UCB score is:
    score = -mu + beta * sigma
    
    Higher scores indicate candidates that are either predicted to be good (low mu)
    or have high uncertainty (high sigma), encouraging exploration.
    
    mu: (n,) array of GP mean predictions
    sigma: (n,) array of GP standard deviation predictions
    f_best: current best observed value (minimum)
    
    Returns: (n,) array of scores, higher is better
    """
    # Avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB parameter. Tuned for 6D Hartmann with ~30 evaluations.
    # Beta balances exploitation (low mu) and exploration (high sigma).
    beta = 1.0
    
    # Score for maximization: -mu + beta * sigma
    scores = -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
