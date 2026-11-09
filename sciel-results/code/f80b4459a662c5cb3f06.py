# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Args:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP standard deviations
        f_best: scalar, best observed value so far
        cost: (2000,) array of evaluation costs
    
    Returns:
        (2000,) array of acquisition scores (higher = evaluate next)
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for expected improvement
    delta = f_best - mu
    z = delta / sigma_safe
    
    # Expected Improvement (EI)
    ei = delta * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (it should be, but clip for safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware normalization
    # The cost is in [1, 2]. Using sqrt(cost) provides a moderate penalty
    # that is less aggressive than linear division but still favors cheaper points.
    cost_penalty = np.sqrt(cost)
    
    # Base acquisition: EI per unit cost
    acq = ei / cost_penalty
    
    # Add a small exploration bonus to help escape local minima
    # This encourages visiting high-uncertainty regions if the cost is reasonable
    exploration_weight = 0.1
    acq += exploration_weight * sigma_safe / cost_penalty
    
    # Ensure all values are finite
    acq = np.where(np.isfinite(acq), acq, 0.0)
    
    return acq
# EVOLVE-BLOCK-END
