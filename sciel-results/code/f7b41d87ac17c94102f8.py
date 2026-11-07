"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Probability of Improvement.
    # POI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # We normalize by cost to prefer cheaper points with high probability of improvement.
    # This creates a sharper peak around high-confidence improvements compared to EI.
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)
    
    # Avoid division by zero or negative cost (cost is 1 + mean(x_scaled), so >= 1)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Scale to ensure finite and non-negative
    scores = poi / cost_safe
    
    # Ensure finite output
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
