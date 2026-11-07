"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Expected improvement per unit cost, with numerical safeguards.
    sigma = np.maximum(sigma, 1e-8)
    z = (f_best - mu) / sigma
    # Clip z to avoid overflow in cdf/pdf
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # Ensure non-negative
    ei = np.maximum(ei, 0.0)
    # Cost-aware: divide by cost
    score = ei / np.maximum(cost, 1e-6)
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    return score
# EVOLVE-BLOCK-END
