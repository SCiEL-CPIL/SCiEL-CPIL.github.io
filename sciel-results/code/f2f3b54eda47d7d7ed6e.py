"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement
    with exponential cost suppression.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Clip EI to non-negative (EI should be >= 0 in theory, but numerical issues can cause negatives)
    ei = np.maximum(ei, 0.0)
    
    # Multiplicative cost gate: exp(-k * cost)
    # Calibrate k based on the scale of EI
    # Typical EI values for cosine8 are on the order of 0.1-1.0
    # Cost ranges from 1 to 2
    # k=1.0 means cost=1 gives factor 0.37, cost=2 gives factor 0.14
    k = 1.5
    cost_gate = np.exp(-k * cost)
    
    # Combine: EI multiplied by cost suppression
    scores = ei * cost_gate
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
