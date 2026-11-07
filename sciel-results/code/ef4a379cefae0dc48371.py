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
    combined with an exploration bonus and exploitation boost, divided by cost.
    
    This approach prioritizes points that have a high probability of being
    better than the current best, while penalizing expensive points.
    The exploration bonus helps escape local minima in Rastrigin, and the
    exploitation boost focuses on points near the current best.
    
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
    
    # Exploration bonus: scaled sigma to encourage visiting uncertain regions
    # This helps escape local minima in Rastrigin
    # Coefficient of 0.15 is slightly more aggressive than 0.1
    beta = 0.15
    sigma_norm = np.std(sigma)
    if sigma_norm < 1e-9:
        sigma_norm = 1e-9
    exploration = beta * sigma_safe / sigma_norm
    
    # Exploitation boost: bonus for points where mu is close to f_best
    # This helps refine the search near the current best
    # Use a Gaussian-like kernel centered at f_best
    # Width is proportional to the scale of f_best
    f_scale = max(abs(f_best), 0.1)
    exploitation_width = 0.1 * f_scale
    exploitation = 0.1 * np.exp(-0.5 * ((mu - f_best) / exploitation_width) ** 2)
    
    # Combine PoI, Exploration, and Exploitation
    numerator = poi + exploration + exploitation
    
    # Cost-aware score
    # We divide by cost to get "improvement per unit cost"
    # Use a slightly softer cost normalization
    scores = numerator / (cost + 0.1)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
