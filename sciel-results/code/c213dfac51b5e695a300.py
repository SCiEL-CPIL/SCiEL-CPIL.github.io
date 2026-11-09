"""Rewritten program with improved acquisition function."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) with an
    exploration bonus (UCB-like) that adapts based on the spread of
    posterior uncertainty relative to the best observed value.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) objective value observed so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising to evaluate.
    """
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB term: balance exploitation and exploration
    # Use an adaptive beta that depends on the relative uncertainty
    # When f_best is far from typical mu values, we may be in early stages
    # Use a fixed but well-tuned beta for UCB
    beta = 0.5
    
    ucb = -mu + beta * sigma_safe
    
    # Blend EI and UCB
    # In early iterations, UCB helps explore; later, EI dominates
    # We can blend based on the magnitude of EI relative to UCB spread
    # Simple approach: weighted sum with fixed weights
    # EI tends to be larger when we're confident about improvement
    # UCB is larger in uncertain regions
    
    # Normalize both to comparable scales
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ucb_max = np.max(ucb) if np.max(ucb) > 0 else 1.0
    
    # Use a geometric mean-like combination or weighted sum
    # Let's try: score = EI + small * UCB
    # Or: score = EI * (1 + small * sigma)
    
    # Actually, let's try a different approach: 
    # Use EI but with a modified sigma that accounts for exploration
    # Or use the "probability of improvement" as an additional term
    
    pi = norm.cdf(z)
    
    # Combine: primary EI, secondary PI for exploration
    score = ei + 0.1 * pi * sigma_safe
    
    return score
# EVOLVE-BLOCK-END
