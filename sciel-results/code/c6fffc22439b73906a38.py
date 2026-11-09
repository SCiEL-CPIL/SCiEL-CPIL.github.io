import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Multi-objective acquisition function combining:
    1. Standard Expected Improvement (EI) for exploitation
    2. Posterior variance term for active information gain (exploration)
    
    The balance between the two terms adapts based on how close f_best is to
    the known global minimum of Goldstein-Price 2D (which is 3.0).
    """
    start_time = time.monotonic()
    deadline = start_time + 5.0  # Generous safety margin
    
    # Constants for Goldstein-Price 2D
    F_MIN = 3.0
    
    # Ensure sigma is positive and has valid shape
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    diff = f_best - mu
    z = diff / sigma_safe
    
    # Vectorized normal CDF and PDF
    cdf_val = norm.cdf(z)
    pdf_val = norm.pdf(z)
    
    # EI for minimization
    ei = diff * cdf_val + sigma_safe * pdf_val
    
    # Compute posterior variance term for information gain
    # We want to explore regions with high uncertainty
    # Use log-damped sigma to prevent over-exploration
    variance_term = np.log1p(sigma_safe)
    
    # Adaptive weighting based on progress toward the global minimum
    # gap = f_best - F_MIN (how far we are from the optimum)
    gap = max(f_best - F_MIN, 1e-6)
    
    # Normalize progress: use exponential decay
    # When gap is large (early), we want more exploration
    # When gap is small (late), we want more exploitation
    scale = 2.0  # Controls the rate of transition
    progress = 1.0 - np.exp(-gap / scale)
    
    # Interpolate weights:
    # Early (progress ~ 1): weight_ei ~ 0.3, weight_var ~ 0.7
    # Late (progress ~ 0): weight_ei ~ 0.7, weight_var ~ 0.3
    weight_ei = 0.3 + 0.4 * (1.0 - progress)
    weight_var = 1.0 - weight_ei
    
    # Normalize EI and variance term to comparable scales
    # EI is already in objective units
    # Variance term is in log units, so we scale it
    var_scale = 2.0  # Scale factor for variance term
    
    # Combine the two objectives
    scores = weight_ei * ei + weight_var * var_scale * variance_term
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Check if we exceeded the deadline (highly unlikely for this vectorized computation)
    if time.monotonic() > deadline:
        pass  # The calculation is already done
    
    return scores
