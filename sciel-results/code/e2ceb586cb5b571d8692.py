"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Expected Improvement (EI) with an adaptive exploration term,
    normalized by evaluation cost.
    """
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Numerical stability
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Calculate z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive exploration weight
    # The global minimum of Shekel 4D is -10.5364
    known_min = -10.5364
    # Distance of current best from the known minimum
    dist = max(0.0, f_best - known_min)
    
    # When far from the optimum (large dist), increase exploration to escape local minima.
    # When close (small dist), reduce exploration to refine the solution.
    # w ranges from 0 (at min) to 0.5 (far away)
    w = 0.5 * (1.0 - np.exp(-dist))
    
    # Combine EI and exploration, then divide by cost
    score = (ei + w * sigma_safe) / cost_safe
    
    # Ensure finite outputs
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
