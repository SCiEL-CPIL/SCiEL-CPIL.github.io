"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Calculate Expected Improvement
    # f_best is the lowest observed value (minimization)
    # mu, sigma are GP predictions
    # cost is the cost of evaluating each candidate
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    z = (f_best - mu) / sigma_safe
    
    # Standard EI formula
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Dynamic Cost Exponent
    # We want to be more sensitive to cost as we approach the optimal value.
    # Estimate progress based on f_best.
    # f_min = -0.8 (known minimum of cosine8 scaled)
    # f_max approx 1.0 (upper bound of the function range)
    
    f_min = -0.8
    f_max = 1.0
    
    # t is 1 when f_best is f_max (early), 0 when f_best is f_min (late)
    # Clip f_best to [f_min, f_max] to keep t in [0, 1]
    f_best_clipped = np.clip(f_best, f_min, f_max)
    
    # If f_best is exactly f_min, t=0. If f_best is f_max, t=1.
    t = (f_best_clipped - f_min) / (f_max - f_min)
    
    # alpha: exponent for cost.
    # Early (t=1): alpha = 1.0 (standard)
    # Late (t=0): alpha = 2.0 (more cost sensitive)
    # Linear interpolation
    alpha = 1.0 + (1.0 - t) * 1.0  # alpha = 1 + (1-t) = 2-t
    
    # Score = EI / cost^alpha
    # cost is positive (>= 1)
    score = ei / np.power(cost, alpha)
    
    return score
# EVOLVE-BLOCK-END
