import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Pure Upper Confidence Bound (UCB) for minimization.
    
    Strategy:
    - Maximizes the score: -mu + kappa * sigma
    - High kappa (5.0) aggressively explores regions with high posterior uncertainty (sigma),
      which is crucial for escaping local minima in the multimodal Hartmann 6D landscape
      within the limited 30-evaluation budget.
    
    Args:
        mu: Array of posterior means (2000,)
        sigma: Array of posterior standard deviations (2000,)
        f_best: Current best observed value (scalar)
        
    Returns:
        Array of scores (2000,). Higher score indicates the candidate 
        is more likely to be the next evaluation point.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and handle numerical stability
    # Although sigma should be positive, clamp to 0 to avoid negative exploration bonus if NaN/Inf issues arise
    safe_sigma = np.maximum(sigma, 0.0)
    
    # UCB for minimization:
    # We want to minimize f. The "optimistic" value for f is the lower bound of the CI: mu - kappa * sigma.
    # We select the point with the lowest possible value, i.e., we maximize the negative of the lower bound?
    # Actually, standard UCB for minimization is defined as maximizing: -mu + kappa * sigma.
    # This rewards low mean (exploitation) and high variance (exploration).
    
    kappa = 5.0
    
    # Calculate scores
    scores = -mu + kappa * safe_sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e6)
    
    return scores
