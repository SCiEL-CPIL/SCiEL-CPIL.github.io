import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
# 330s limit. We want to be well before.
_DEADLINE = _START_TIME + 325

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (PoI)
    combined with a small exploration bonus, divided by cost.
    
    This approach prioritizes points that have a high probability of being
    better than the current best, while penalizing expensive points.
    This is often more robust for multimodal functions like Rastrigin
    compared to EI, as it doesn't get misled by large but unlikely improvements.
    
    Args:
        mu (np.ndarray): Mean predictions from the GP, shape (2000,).
        sigma (np.ndarray): Standard deviation predictions from the GP, shape (2000,).
        f_best (float): The lowest observed function value.
        cost (np.ndarray): Cost of evaluating each candidate, shape (2000,).
    
    Returns:
        np.ndarray: Scores for the candidates, shape (2000,). Higher is better.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: Prioritize cheap points if we are running out of time
        return 1.0 / (cost + 1e-9)

    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score for the probability of improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement (PoI) for minimization
    # P(f(x) < f_best) = Phi(z)
    poi = norm.cdf(z)
    
    # Add a small exploration bonus to encourage visiting uncertain regions
    # This helps escape local minima in Rastrigin.
    # We use a scaled version of sigma to keep the bonus small relative to PoI (which is in [0,1])
    # A coefficient of 0.1 is conservative.
    beta = 0.1
    exploration = beta * sigma_safe / (np.std(sigma) + 1e-9)
    
    # Combine PoI and Exploration
    # We want the score to be dominated by PoI but slightly boosted by uncertainty
    numerator = poi + exploration
    
    # Cost-aware score
    # We divide by cost to get "improvement per unit cost"
    scores = numerator / (cost + 1e-9)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
