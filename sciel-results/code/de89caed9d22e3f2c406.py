"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma
    
    # Calculate standard Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add an exploration bonus to encourage sampling of uncertain regions
    # This keeps the design within the EI family but makes it less greedy
    beta = 0.5
    acquisition_value = ei + beta * sigma
    
    # Cost-aware score: maximize improvement per unit cost
    # cost is the cost of evaluating each candidate
    score = acquisition_value / cost
    
    return score
# EVOLVE-BLOCK-END
