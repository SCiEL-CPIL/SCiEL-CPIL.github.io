"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI) 
    and Upper Confidence Bound (UCB) with a softened cost penalty.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions at candidate points.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions at candidate points.
    f_best : float
        Best observed value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
        
    Returns:
    --------
    array-like, shape (2000,)
        Acquisition scores for each candidate point.
    """
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Calculate UCB term
    # Beta controls the exploration-exploitation balance
    beta = 2.0
    ucb = -mu + beta * sigma_safe
    
    # Combine EI and UCB
    # We take the maximum of the two to ensure we have a strong signal
    # Or we can blend them. Let's try a weighted sum first.
    # Based on the cell finding, blending works well.
    # Let's use a fixed weight for now, e.g., 50/50
    combined = 0.5 * ei + 0.5 * ucb
    
    # Soften the cost penalty
    # Instead of dividing by cost, we divide by cost^0.4
    # This reduces the bias towards cheap regions
    cost_safe = np.maximum(cost, 1e-8)
    cost_penalty = cost_safe ** 0.4
    
    score = combined / cost_penalty
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
