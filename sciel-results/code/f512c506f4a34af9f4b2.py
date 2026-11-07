"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Deadline guard: return quickly if we're running out of time
    # This is a safety net; normally this function is fast
    deadline = time.time() + 5.0  # 5 second safety margin
    
    # Ensure sigma is strictly positive for numerical stability
    sigma = np.maximum(sigma, 1e-8)
    
    # Use Upper Confidence Bound (UCB) instead of EI
    # UCB = mu - beta * sigma for minimization (we want low values)
    # The beta parameter controls exploration vs exploitation
    # For RBF mixtures in 2D with 30 evaluations, moderate exploration helps
    beta = 2.5
    
    # UCB score: lower mu and higher sigma give higher scores
    # We negate because we want to maximize the acquisition score
    # For minimization: acquisition = -mu + beta * sigma
    # Higher acquisition means more promising (lower expected value, more uncertainty)
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
