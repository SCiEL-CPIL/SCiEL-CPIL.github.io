import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Weighted Upper Confidence Bound (CW-UCB).
    
    Args:
        mu: Array of shape (2000,) with predicted means in objective units.
        sigma: Array of shape (2000,) with predicted standard deviations in objective units.
        f_best: Scalar, the lowest observed value so far.
        cost: Array of shape (2000,) with the cost of evaluating each candidate.
        
    Returns:
        Array of shape (2000,) with acquisition scores (higher is better).
    """
    # Deadline guard
    t_start = time.time()
    
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Cost-Weighted UCB
    # For minimization, we maximize the upper confidence bound of the regret,
    # which is proportional to (kappa * sigma - mu).
    # We divide by cost to prioritize cheaper evaluations.
    kappa = 3.0
    
    scores = (kappa * sigma - mu) / cost
    
    # Safety check for time (should never trigger for 2000 points)
    if time.time() - t_start > 290:
        # Fallback: Exploit best mean, penalized by cost
        scores = -mu / cost
        
    return scores
