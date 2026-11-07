import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using expected improvement with adaptive cost penalty.
    
    Uses a smooth cost-aware formulation that balances expected improvement
    against evaluation cost, with adaptive weighting based on cost distribution.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute z-score for expected improvement
    z = (f_best - mu) / sigma_safe
    
    # Use a smoother clipping to avoid discontinuities
    z_clipped = np.clip(z, -10, 10)
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_val = norm.cdf(z_clipped)
    pdf_val = norm.pdf(z_clipped)
    
    ei = (f_best - mu) * cdf_val + sigma * pdf_val
    
    # Adaptive cost-aware scoring: use a power-weighted cost penalty
    # This allows the algorithm to be more sensitive to cost differences
    # when costs are tightly clustered, and less sensitive when spread out
    cost_min = np.min(cost_safe)
    cost_max = np.max(cost_safe)
    cost_range = cost_max - cost_min + 1e-10
    
    # Normalize cost to [0, 1] for adaptive weighting
    cost_norm = (cost_safe - cost_min) / cost_range
    
    # Use a smooth cost penalty that emphasizes low-cost points
    # The exponent controls how aggressively we penalize high-cost points
    # A value of 1.0 gives linear scaling, >1.0 is more aggressive
    cost_penalty = np.power(cost_norm + 1e-6, 1.2)
    
    # Combine EI with cost-aware penalty
    # This formulation rewards points with high EI and low cost
    scores = ei / (1.0 + cost_penalty * 0.5)
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
