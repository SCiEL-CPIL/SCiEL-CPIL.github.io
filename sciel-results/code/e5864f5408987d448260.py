"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Ensure sigma is strictly positive for numerical stability
    sigma = np.maximum(sigma, 1e-8)
    
    # Standard Expected Improvement for minimization
    z = (f_best - mu) / sigma
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # Clip to non-negative (EI should be >= 0)
    ei = np.maximum(ei, 0.0)
    
    # Add a small exploration bonus proportional to sigma
    # This helps escape local optima in the RBF mixture
    # Scale the exploration term to be small relative to EI
    # Use a fixed small coefficient since we don't have iteration count
    exploration_bonus = 0.05 * sigma
    
    # Combine: EI dominates, exploration provides tie-breaking in flat regions
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
