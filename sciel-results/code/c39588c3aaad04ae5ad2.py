"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points.
    
    Uses a combination of Expected Improvement (EI) and Upper Confidence Bound (UCB)
    to balance exploitation and exploration. The UCB component helps explore
    regions of high uncertainty which is crucial for multi-modal functions like
    Styblinski-Tang.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising to evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Expected Improvement component
    # For minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Upper Confidence Bound component for exploration
    # UCB = mu - kappa * sigma (for minimization, we want low mu and high sigma)
    # We use a kappa that balances exploration
    kappa = 1.5
    ucb = mu - kappa * sigma_safe
    
    # Normalize both components to have similar scale
    # EI is typically positive, UCB can be negative
    # We'll combine them with a weighted sum
    
    # Scale EI to be comparable to UCB
    # The typical magnitude of EI is around sigma, while UCB is around mu
    # We want to balance exploitation (EI) and exploration (UCB)
    
    # A robust combination: use EI primarily but add a small UCB boost
    # to encourage exploration in uncertain regions
    score = ei + 0.1 * (-sigma_safe)  # Small penalty for high uncertainty in EI context
    
    # Actually, let's use a cleaner approach:
    # Primary: EI for exploitation
    # Secondary: Add exploration bonus for points with high sigma and low mu
    
    # Better approach: Use a modified EI that accounts for exploration
    # or use a simple weighted combination of normalized EI and UCB
    
    # Let's compute normalized versions
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ei_norm = ei / (ei_max + 1e-10)
    
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min + 1e-10
    ucb_norm = (ucb - ucb_min) / ucb_range
    
    # Invert UCB norm since lower UCB is better for minimization
    ucb_norm_inv = 1.0 - ucb_norm
    
    # Weighted combination: 70% EI, 30% inverted UCB
    score = 0.7 * ei_norm + 0.3 * ucb_norm_inv
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
