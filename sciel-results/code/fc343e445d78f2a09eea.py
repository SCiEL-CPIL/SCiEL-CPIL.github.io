"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Expected Improvement with cost discounting.
    
    Uses the Expected Improvement (EI) criterion, which is known to be
    more robust than UCB for multi-modal functions. The EI value is
    discounted by the evaluation cost to favor cheaper evaluations.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - lowest observed value
    cost : array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
    --------
    score : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    t0 = time.time()
    
    # Safety check for empty or invalid inputs
    if len(mu) == 0:
        return np.array([])
        
    # Ensure sigma is strictly positive to avoid division by zero or log issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    # Standardize the improvement
    z = (f_best - mu) / sigma_safe
    
    # Compute Phi(z) and phi(z)
    # Using scipy.stats.norm for accurate computation
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Clip EI to be non-negative (EI should be >= 0 by definition)
    ei = np.maximum(ei, 0.0)
    
    # Cost penalty: discount by cost to favor cheaper evaluations
    # cost is in [1, 2], so 1/cost is in [0.5, 1.0]
    cost_safe = np.maximum(cost, 1.0)
    score = ei / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline guard
    elapsed = time.time() - t0
    if elapsed > 250:
        # Just return what we have; operations are fast
        pass
        
    return score
    # EVOLVE-BLOCK-END
