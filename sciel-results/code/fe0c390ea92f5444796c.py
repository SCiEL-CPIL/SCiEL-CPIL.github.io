# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: the evaluator kills after 330s. We use a safe margin.
_DEADLINE = time.time() + 320.0  # 320s to be safe

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions in objective units.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x scaled to [0,1])).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
        # Return a safe default: exploit best
        return -mu / (cost + 1e-9)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware: divide by cost
    # But we want to balance exploration vs exploitation
    # Use a slightly modified approach:
    # - For high sigma (exploration), we might want to give more credit
    # - For low sigma (exploitation), we want to focus on high EI
    
    # Adaptive approach: weight exploration based on how far we are from optimum
    # If f_best is still high, explore more; if f_best is low, exploit more
    
    # Simple cost-aware EI
    scores = ei / cost
    
    # Add a small exploration bonus for high uncertainty
    # This helps escape local minima
    exploration_bonus = 0.05 * sigma / cost
    
    scores = scores + exploration_bonus
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
