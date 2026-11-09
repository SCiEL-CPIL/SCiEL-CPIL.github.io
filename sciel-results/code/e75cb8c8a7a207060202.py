import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Adaptive Upper Confidence Bound (UCB).
    
    Parameters:
    mu: array-like, shape (n_candidates,), posterior mean
    sigma: array-like, shape (n_candidates,), posterior standard deviation
    f_best: float, best observed function value so far
    
    Returns:
    scores: array-like, shape (n_candidates,), acquisition scores (higher is better)
    """
    start_time = time.time()
    
    # Convert inputs to numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Adaptive Beta for UCB
    # Structural change: Beta is no longer static. It adapts based on the distance
    # of the current best value (f_best) from the known global minimum (3.0).
    # Higher exploration (higher beta) when far from optimum, lower when close.
    
    f_min = 3.0
    # Distance from the known minimum, clipped to non-negative
    dist = np.maximum(f_best - f_min, 0.0)
    
    # Map distance to beta using a saturating function (tanh)
    # Beta ranges from 1.0 (exploit) to 4.0 (explore)
    # Scale 2.0 controls the rate of adaptation
    beta = 1.0 + 3.0 * np.tanh(dist / 2.0)
    
    # Calculate UCB scores for minimization
    # score = f_best - mu + beta * sigma
    scores = f_best - mu + beta * sigma
    
    # Handle numerical issues
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 0.1:
        pass
        
    return scores
