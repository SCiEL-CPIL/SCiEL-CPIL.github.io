"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement with
    a variance-aware exploration term, normalized by evaluation cost.
    
    The acquisition score is:
      (EI + beta * sigma) / cost
    
    where EI is the expected improvement over f_best and beta controls
    the exploration-exploitation tradeoff. The cost normalization penalizes
    expensive evaluations.
    
    Parameters:
    mu: (2000,) array of GP mean predictions
    sigma: (2000,) array of GP standard deviations
    f_best: float, the lowest observed function value
    cost: (2000,) array of evaluation costs
    
    Returns:
    scores: (2000,) array of acquisition scores (higher is better)
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # For minimization, we want to improve (decrease) f_best.
    # EI is the expected amount by which the function value at x
    # is below f_best, assuming Y ~ N(mu, sigma^2).
    
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues with very large/small values
    z = np.clip(z, -10, 10)
    
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    # Expected Improvement for minimization
    EI = (f_best - mu) * Phi + sigma_safe * phi
    
    # Ensure EI is non-negative (numerical safety)
    EI = np.maximum(EI, 0.0)
    
    # Exploration term: reward high uncertainty regions
    # This helps escape local optima in the 8d landscape
    beta = 0.1
    exploration = beta * sigma_safe
    
    # Combined acquisition: EI + exploration
    acquisition_score = EI + exploration
    
    # Cost-aware normalization: divide by cost
    # cost is in [1, 2] for x in [-1, 1]^8
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = acquisition_score / cost_safe
    
    # Handle any non-finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
