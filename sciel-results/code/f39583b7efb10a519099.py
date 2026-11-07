"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling acquisition function.
    
    Draws a sample from the posterior at each candidate point:
      sample = mu + sigma * z, where z ~ N(0, 1)
    Then scores by -sample / cost (lower sampled value per cost is better).
    Returns higher scores for candidates likely to yield better (lower) objective
    values per unit cost.
    """
    # Deadline guard: track elapsed time
    t0 = time.time()
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    # Draw a single standard normal sample for each candidate
    z = np.random.standard_normal(mu.shape)
    
    # Posterior sample: mu + sigma * z
    sample = mu + sigma * z
    
    # Score: we want to maximize (i.e., pick the point with lowest sampled value per cost)
    # So score = -sample / cost (higher is better for the evaluator which picks max)
    scores = -sample / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Deadline check: if we've used too much time, just return what we have
    if time.time() - t0 > 295:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
