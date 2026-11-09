# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB).
    
    For minimization, we maximize:
    score = -mu + beta * sigma
    
    This balances exploitation (low mu) and exploration (high sigma).
    UCB is structurally different from Expected Improvement (EI) as it
    uses a linear additive term for uncertainty rather than an integral
    over the improvement distribution.
    
    Parameters:
    mu: Mean of the GP posterior (shape 2000,)
    sigma: Standard deviation of the GP posterior (shape 2000,)
    f_best: Best observed function value so far
    
    Returns:
    scores: Array of acquisition scores (shape 2000,), higher is better
    """
    # Ensure sigma is non-negative and handle potential numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # UCB parameter (exploration rate)
    # A value of 2.0 is a common choice for GP-UCB
    beta = 2.0
    
    # Calculate UCB score for minimization
    # We want to minimize f(x), so we prefer low mu.
    # We want to explore high sigma.
    # score = -mu + beta * sigma
    score = -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
