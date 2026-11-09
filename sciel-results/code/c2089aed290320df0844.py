"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Xi-Expected Improvement.
    
    Args:
        mu: (2000,) array of GP means.
        sigma: (2000,) array of GP standard deviations.
        f_best: float, the best observed function value so far.
        cost: (2000,) array of evaluation costs.
        
    Returns:
        (2000,) array of acquisition scores (higher is better).
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Exploration parameter xi
    # A small xi encourages finding improvements, while not ignoring small ones entirely
    xi = 0.005
    
    # Calculate z-score for EI with xi
    delta = f_best - mu - xi
    z = delta / sigma_safe
    
    # Expected Improvement (EI) formula for minimization
    # EI = (f_best - xi - mu) * Phi(z) + sigma * phi(z)
    ei = delta * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Numerical stability: clamp negative values to 0
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring
    # Using sqrt(cost) softens the penalty for expensive evaluations compared to linear cost,
    # which may help explore the center of the domain (moderate cost) more effectively
    # if the GP predicts high uncertainty or low mean there.
    score = ei / np.sqrt(cost)
    
    # Ensure all outputs are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
