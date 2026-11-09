"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for the deadline guard
_start_time = time.time()
_DEADLINE_SECONDS = 290.0  # Safety margin before 330s limit

def _elapsed():
    return time.time() - _start_time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Combines Expected Improvement (EI) with a cost penalty,
    plus a small exploration bonus to avoid premature convergence.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical instability
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate z-score
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware acquisition: EI / cost
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-8)
    
    # Base score: EI normalized by cost
    base_score = ei / cost
    
    # Add a small exploration bonus proportional to sigma / cost
    # This helps escape local minima by giving some weight to uncertain regions
    # The bonus is small (0.1x) to not dominate the EI signal
    exploration_bonus = 0.1 * sigma / cost
    
    scores = base_score + exploration_bonus
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # If we're close to the deadline, return a simpler score to save time
    if _elapsed() > _DEADLINE_SECONDS:
        # Fall back to just EI/cost without the bonus computation
        scores = np.nan_to_num(base_score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
