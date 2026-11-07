"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-Effective Probability of Improvement (cPoI).
    # Structural change from Expected Improvement (EI) to Probability of Improvement (PoI).
    # PoI measures the probability that f(x) < f_best, which can provide a different
    # exploration/exploitation balance compared to EI, potentially escaping local
    # optima or handling noise differently.
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Z-score
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best)
    poi = norm.cdf(z)
    
    # Cost-aware acquisition: PoI divided by evaluation cost
    scores = poi / cost
    
    # Ensure all scores are finite (safety check)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
