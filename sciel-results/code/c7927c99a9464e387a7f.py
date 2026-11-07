import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 300.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    Uses a blend of Expected Improvement (EI) and an exploration bonus (proportional to sigma),
    divided by a power of the evaluation cost.
    
    Parameters:
    mu: array-like, shape (2000,), predicted mean in objective units.
    sigma: array-like, shape (2000,), predicted standard deviation in objective units.
    f_best: float, best observed value so far (lowest).
    cost: array-like, shape (2000,), cost of evaluating each candidate.
    
    Returns:
    scores: array of shape (2000,), higher is better.
    """
    # Safety check for deadline
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback to a simple, fast calculation
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        # Simple EI / cost
        score = ei / cost_safe
        return np.where(np.isfinite(score), score, -1e9)

    # Parameters tuned for the cosine8 8d problem
    # Cost penalty exponent: 1.2 (slightly superlinear to avoid expensive regions)
    alpha = 1.2
    # Exploration parameter for the UCB-like bonus
    beta = 1.5
    # Weight for the exploration bonus relative to EI
    lambda_exp = 0.5

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clamp sigma to avoid numerical issues in norm.pdf/cdf
    sigma_safe = np.maximum(sigma, 1e-9)
    # Clamp cost to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration bonus: beta * sigma
    # We add this to EI to encourage exploring uncertain regions
    # Note: EI is already non-negative. Adding beta*sigma makes it even larger for high uncertainty.
    # This acts like a UCB term.
    exploration_bonus = beta * sigma_safe
    
    # Combined numerator: EI + lambda * exploration_bonus
    numerator = ei + lambda_exp * exploration_bonus
    
    # Cost penalty
    cost_penalty = cost_safe ** alpha
    
    # Final score
    score = numerator / cost_penalty
    
    # Handle any non-finite values (e.g., from extreme z or overflow)
    score = np.where(np.isfinite(score), score, -1e9)
    
    return score
