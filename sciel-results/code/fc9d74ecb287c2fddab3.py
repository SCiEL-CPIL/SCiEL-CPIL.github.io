"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Pure Expected Improvement per unit cost.
    # This matches the top-performing inspiration (score 1.0000).
    # The current program's exploration bonus (0.05 * sigma) likely caused
    # slight over-exploration, reducing the score to 0.8783.
    # UCB (inspiration 3) and larger exploration bonuses (inspiration 2)
    # performed worse, confirming that pure EI/cost is the strongest structural choice.
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard normal CDF and PDF for Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware acquisition: EI divided by evaluation cost
    scores = ei / cost
    
    # Ensure all scores are finite (safety check)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
