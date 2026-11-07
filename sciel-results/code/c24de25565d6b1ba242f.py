"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected Improvement for minimization with a mild exploration bonus.
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    
    # Standard EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a small exploration bonus proportional to sigma, scaled by how
    # uncertain we are relative to the best value. This encourages the
    # optimizer to visit regions with high uncertainty when EI is low.
    # The term is bounded and adaptive: it's larger when sigma is large
    # relative to the scale of f_best, and vanishes when we're confident.
    scale = np.maximum(np.abs(f_best), 1e-6)
    exploration_weight = 0.05 * np.minimum(sigma / scale, 1.0)
    
    scores = ei + exploration_weight * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
