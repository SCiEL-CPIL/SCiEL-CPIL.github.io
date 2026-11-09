"""Acquisition function using Probability-of-Improvement with Entropy Bonus."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining probability of improvement with entropy bonus.
    
    For minimization, we want to find points likely to have f(x) < f_best.
    Score = P(f(x) < f_best) * (f_best - mu) + beta * sigma
    
    This is distinct from EI (which is E[max(0, f_best - f(x))]) because:
    - It uses the probability of improvement as a weight on the expected improvement
    - It adds a direct uncertainty bonus scaled by sigma
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array of shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        The lowest observed value so far.
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-score for improvement: z = (f_best - mu) / sigma
    # If mu < f_best, z > 0, meaning high probability of improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of improvement: P(f(x) < f_best) = Phi(z)
    # This is a smooth, bounded function in [0, 1]
    pi_prob = norm.cdf(z)
    
    # Expected improvement magnitude: f_best - mu (can be negative if mu > f_best)
    # Clip to non-negative for the "expected gain if improved" interpretation
    expected_gain = np.maximum(f_best - mu, 0.0)
    
    # Weighted improvement: probability of improvement times expected gain
    # This peaks when both probability and gain are high
    weighted_improvement = pi_prob * expected_gain
    
    # Entropy/uncertainty bonus: encourage exploration in high-variance regions
    # Scale the bonus relative to the range of sigma to make it adaptive
    sigma_range = np.max(sigma_safe) - np.min(sigma_safe)
    if sigma_range < 1e-8:
        sigma_range = 1e-8
    
    # Normalize sigma to [0, 1] for adaptive scaling
    sigma_norm = (sigma_safe - np.min(sigma_safe)) / sigma_range
    
    # Beta controls exploration-exploitation balance
    # Use a moderate value that provides meaningful exploration without dominating
    beta = 0.5
    
    # Final score: weighted improvement + entropy bonus
    scores = weighted_improvement + beta * sigma_norm
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
