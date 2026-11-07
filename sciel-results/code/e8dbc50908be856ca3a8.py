"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI) for minimization.
    
    EI measures the expected improvement over the current best (f_best).
    For minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    We divide by cost to get improvement per unit cost.
    
    Parameters:
    mu: numpy array of shape (N,) - GP mean in objective units.
    sigma: numpy array of shape (N,) - GP standard deviation in objective units.
    f_best: float - Best observed value so far.
    cost: numpy array of shape (N,) - Cost of evaluating each candidate.
    
    Returns:
    numpy array of shape (N,) - Acquisition scores. Higher is better.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Standardize the improvement
    z = (f_best - mu) / sigma
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware: divide by cost
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    scores = ei / cost
    
    # Handle any numerical instabilities
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
