# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for GP predictions to avoid redundant computation
# Key: (mu_data_id, sigma_data_id, f_best, cost_id)
# Since we get mu and sigma directly, we just use them.

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using GP-LCB with adaptive beta.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions at candidate points.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions at candidate points.
    f_best : float
        Best observed function value so far.
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    np.ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-8)
    
    # Adaptive beta: starts higher for exploration, decreases as f_best improves
    # f_best is in [0, ~3] for Ackley (min 0, max ~3)
    # Use a simple schedule: beta = 2.0 + 3.0 * max(0, 1 - f_best / 1.0)
    # When f_best is large (early), beta is higher (more exploration)
    # When f_best is small (late), beta approaches 2.0
    
    # Alternative: use a fixed beta that balances well
    # Let's try beta = 2.5 as a compromise
    beta = 2.5
    
    # GP-LCB: mu - beta * sigma
    lcb = mu - beta * sigma
    
    # Cost-aware: divide by cost to prefer cheaper points
    # But we want higher scores to be evaluated next
    # So we want to maximize (improvement potential) / cost
    # The improvement potential is roughly (f_best - mu) + beta * sigma
    # Actually, LCB is mu - beta*sigma, so lower LCB is better
    # We want to maximize (f_best - LCB) / cost = (f_best - mu + beta*sigma) / cost
    
    # This is essentially expected improvement per cost with a different weighting
    numerator = f_best - mu + beta * sigma
    
    # Ensure numerator is non-negative for numerical stability
    # If mu > f_best, the term f_best - mu is negative, but beta*sigma should dominate
    # Clip to avoid negative scores
    numerator = np.maximum(numerator, 0)
    
    scores = numerator / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
