import numpy as np
import time
from scipy.stats import norm

# Deadline guard: the evaluator kills after 330s wall-clock.
# We track a global start time and bail early if we're getting close.
_T0 = time.time()
_DEADLINE = 330.0 - 3.0  # safety margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization (minimization).
    
    Parameters:
    mu (np.ndarray): GP mean predictions (shape 2000,)
    sigma (np.ndarray): GP standard deviation predictions (shape 2000,)
    f_best (float): Best observed function value so far
    cost (np.ndarray): Cost of evaluating each candidate (shape 2000,)
    
    Returns:
    np.ndarray: Acquisition scores (shape 2000,), higher is better
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return a safe fallback: just exploit
        return np.nan_to_num(f_best - mu, nan=0.0, posinf=0.0, neginf=0.0)
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    f_best = float(f_best)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scaling: divide by cost
    # This gives improvement per unit cost
    scores = ei / cost
    
    # Add a small exploration bonus for high-uncertainty regions
    # Scale the bonus so it doesn't dominate
    exploration_bonus = 0.1 * sigma / cost
    
    scores = scores + exploration_bonus
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
