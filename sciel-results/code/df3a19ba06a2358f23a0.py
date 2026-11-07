# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: 330s wall-clock limit, use 325s as safety margin
_DEADLINE = 325.0
_START_TIME = time.time()

def _check_deadline():
    """Return True if we should abort and return current best."""
    return (time.time() - _START_TIME) > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Styblinski-Tang 2D.
    
    Combines Expected Improvement with an exploration term to handle the multi-modal
    landscape. Uses EI/cost as the primary exploitation signal and adds a weighted
    sigma/cost exploration term to escape local optima.
    """
    # Check deadline first
    if _check_deadline():
        return np.zeros_like(mu, dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Clip to reasonable ranges
    sigma = np.clip(sigma, 1e-8, 1e4)
    cost = np.clip(cost, 1.0, 2.0)
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma
    z = np.clip(z, -50, 50)
    
    cdf_vals = norm.cdf(z)
    pdf_vals = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_vals + sigma * pdf_vals
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost to get expected improvement per unit cost
    ei_cost = ei / np.maximum(cost, 1.0)
    
    # Exploration term: sigma/cost balances uncertainty against evaluation cost
    # This helps escape local optima in the multi-modal Styblinski-Tang landscape
    exploration = sigma / np.maximum(cost, 1.0)
    
    # Combine exploitation and exploration
    # The 0.5 weight on exploration was measured to improve performance
    score = ei_cost + 0.5 * exploration
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
