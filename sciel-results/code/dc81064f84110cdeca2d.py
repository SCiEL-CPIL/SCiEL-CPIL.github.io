"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Standard EI components
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Expected Improvement (minimization perspective)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure EI is finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 290.0:
        # Fallback to simple EI/cost
        scores = ei / cost
        return np.where(np.isfinite(scores), scores, -np.inf)
    
    # Global budget-aware scaling strategy
    # 1. Rank the Expected Improvement values to get a normalized score in [0, 1]
    #    This makes the acquisition function invariant to the scale of EI, which can vary
    #    significantly as the GP updates.
    ei_ranks = np.argsort(np.argsort(ei)).astype(float)
    n = len(ei)
    ei_norm = ei_ranks / (n - 1)  # Now in [0, 1]
    
    # 2. Rank the costs to get a normalized cost percentile in [0, 1]
    cost_ranks = np.argsort(np.argsort(cost)).astype(float)
    cost_norm = cost_ranks / (n - 1)  # Now in [0, 1]
    
    # 3. Construct the final score
    # We want to maximize EI and minimize cost.
    # The penalty weight determines how strongly we prefer cheaper points when EI is similar.
    # A weight of 0.5 means that a point with the same EI rank but at the 90th percentile of cost
    # will be penalized by 0.5 * 0.9 = 0.45, which is significant compared to the EI range [0, 1].
    # This effectively creates a "budget-aware" selection where we only pick expensive points
    # if they offer significantly higher expected improvement.
    
    penalty_weight = 0.5
    scores = ei_norm - penalty_weight * cost_norm
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
