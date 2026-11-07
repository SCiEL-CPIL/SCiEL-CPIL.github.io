import numpy as np
from scipy.stats import norm
import time

# Global deadline tracking
_acq_start_time = None
_DEADLINE = 300.0  # Safety margin before the 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the Gaussian Process at candidate points.
    sigma : array-like, shape (2000,)
        Standard deviation of the Gaussian Process at candidate points.
    f_best : float
        Best observed function value so far (minimum).
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : array, shape (2000,)
        Acquisition scores. Higher scores are better.
    """
    global _acq_start_time
    if _acq_start_time is None:
        _acq_start_time = time.time()
    
    # Deadline guard: if we're too close to the deadline, return a simple fallback
    elapsed = time.time() - _acq_start_time
    if elapsed > _DEADLINE - 5.0:
        # Fallback: just use -mu/cost to be safe
        mu = np.asarray(mu, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        cost = np.maximum(cost, 1e-8)
        score = -mu / cost
        score = np.where(np.isfinite(score), score, -1e10)
        return score
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases for sigma to avoid division by zero or log(0)
    sigma = np.maximum(sigma, 1e-8)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-8)
    
    # Calculate Expected Improvement (EI) for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware EI: divide by cost to penalize expensive evaluations
    # Use a power of cost to make the penalty stronger
    cost_ei = ei / (cost ** 1.5)
    
    # Additional exploration term: sigma / cost
    # This encourages evaluating high-uncertainty regions
    exploration = sigma / cost
    
    # Combine: primarily EI-driven, with a small exploration bonus
    # Weight the exploration term to be modest
    score = cost_ei + 0.1 * exploration
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
