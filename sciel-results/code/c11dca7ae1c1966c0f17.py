# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Adaptive LCB Acquisition Function.
    
    Structure:
    1. Uses Lower Confidence Bound (LCB) for minimization: -mu + beta * sigma.
    2. Beta is adaptive based on the current mean uncertainty (sigma).
       Early in the run (high sigma), beta is larger to encourage exploration.
       Late in the run (low sigma), beta is smaller to encourage exploitation.
    3. The score is divided by the cost of the candidate to prioritize
       cost-effective improvements (Optimistic Gain per Unit Cost).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety checks
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    mu = np.where(np.isfinite(mu), mu, 0.0) # Fallback to 0 if invalid, though f_best should guide this
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Adaptive Beta:
    # mean_sigma represents the current level of uncertainty.
    # In Hartmann 3D with standardized targets, early sigma ~ 1.0, late sigma ~ 0.1-0.5.
    # We want beta to scale with this.
    # Base beta ~ 2.0 for standard BO.
    # If mean_sigma is high, increase beta. If low, decrease.
    mean_sigma = np.mean(sigma)
    
    # Linear scaling: beta = base + slope * mean_sigma
    # If mean_sigma = 1.0, beta = 3.0 (Explore more)
    # If mean_sigma = 0.2, beta = 1.4 (Exploit more)
    beta = 1.0 + 2.0 * mean_sigma
    
    # LCB Score: -mu + beta * sigma
    lcb_scores = -mu + beta * sigma
    
    # Cost-Aware Normalization:
    # We want to maximize (Potential Improvement) / Cost.
    # LCB is an optimistic bound on improvement.
    scores = lcb_scores / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
