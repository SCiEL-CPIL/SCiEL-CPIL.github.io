import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Expected Improvement
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Variance-aware exploration boost (Knowledge Gradient-inspired)
    # This encourages exploring high-uncertainty regions to escape local optima
    exploration_boost = 1.0 * sigma
    
    return ei + exploration_boost
