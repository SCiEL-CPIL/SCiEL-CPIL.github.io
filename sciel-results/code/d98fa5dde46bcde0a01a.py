import numpy as np
import time
from scipy.stats import norm

# Global timer for deadline guard
_start_time = time.time()
_DEADLINE = 300.0  # Safety margin from 330s kill

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function: Expected Improvement + Exploration Term.
    
    This function computes a score for each candidate point where higher scores
    indicate more promising candidates for evaluation. It combines Expected Improvement
    (EI), which balances exploitation and exploration based on the current best value,
    with an explicit exploration term proportional to the posterior standard deviation.
    
    Parameters:
    mu: array of shape (N,) - posterior mean
    sigma: array of shape (N,) - posterior standard deviation
    f_best: scalar - best observed value so far (lower is better)
    
    Returns:
    scores: array of shape (N,) - acquisition scores (higher is better)
    """
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Return a valid array of zeros or something finite to avoid crash
        # But the evaluator kills us anyway. Just return fast.
        return np.zeros_like(mu)

    # Handle numerical stability
    # Ensure sigma is non-negative and not zero to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # Since we want to minimize, improvement is f_best - f(x).
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical errors can make it slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # Exploration Term:
    # Add a term proportional to sigma to encourage exploration in uncertain regions.
    # This acts like a UCB (Upper Confidence Bound) for minimization:
    # We prefer points with low predicted value (low mu) and high uncertainty (high sigma).
    # EI already favors low mu, but adding sigma explicitly boosts exploration.
    # Weight 0.5 is a heuristic balance found effective in similar landscapes.
    exploration_term = 0.5 * sigma_safe
    
    # Combine EI and Exploration
    scores = ei + exploration_term
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
