import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP standard deviation predictions
    - f_best: float - best observed function value so far
    - cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    - scores: array of shape (2000,) - acquisition scores (higher = better)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clamp sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues with extreme values
    z = np.clip(z, -50, 50)
    
    # Compute EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost to get improvement per unit cost
    # But also add a small exploration bonus for high uncertainty
    # The cost is 1 + mean(x), ranging from 1 to 2
    
    # Primary score: EI per unit cost
    score = ei / np.maximum(cost, 1e-9)
    
    # Add a small exploration bonus weighted by uncertainty and cost efficiency
    # This helps explore uncertain regions that are cheap
    # Normalize sigma to [0, 1] for the bonus
    sigma_max = np.max(sigma_safe) if len(sigma_safe) > 0 else 1.0
    sigma_norm = sigma_safe / (sigma_max + 1e-9)
    
    # Exploration bonus: favors high uncertainty, low cost
    exploration_bonus = 0.1 * sigma_norm / np.maximum(cost, 1e-9)
    
    # Combine
    scores = score + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
