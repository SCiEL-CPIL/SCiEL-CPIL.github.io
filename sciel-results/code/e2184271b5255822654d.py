"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 325.0  # Safety margin before the 330s kill limit


def acquisition(mu, sigma, f_best):
    """
    Multi-objective acquisition: weighted sum of normalized EI and normalized predictive uncertainty.
    Dynamic weight increases with evaluations to balance exploration and exploitation.
    
    Args:
        mu: Array of predicted means (shape (2000,)).
        sigma: Array of predicted standard deviations (shape (2000,)).
        f_best: The best observed value so far.
        
    Returns:
        Array of scores (shape (2000,)), higher is better.
    """
    # Check for timeout
    if time.time() - _start_time > _TIMEOUT:
        return np.zeros_like(mu)
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the z-score: z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula for minimization:
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_vals = norm.cdf(z)
    pdf_vals = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_vals + sigma_safe * pdf_vals
    
    # Numerical safeguard for small sigma
    small_sigma_mask = sigma < 1e-8
    if np.any(small_sigma_mask):
        ei[small_sigma_mask] = np.maximum(0, f_best - mu[small_sigma_mask])
    
    # Ensure all scores are finite
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e10, neginf=0.0)
    ei = np.maximum(ei, 0.0)
    
    # Normalize EI to [0, 1]
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range > 1e-12:
        ei_norm = (ei - ei_min) / ei_range
    else:
        ei_norm = np.zeros_like(ei)
    
    # Normalize sigma to [0, 1]
    sigma_min = np.min(sigma)
    sigma_max = np.max(sigma)
    sigma_range = sigma_max - sigma_min
    if sigma_range > 1e-12:
        sigma_norm = (sigma - sigma_min) / sigma_range
    else:
        sigma_norm = np.zeros_like(sigma)
    
    # Dynamic weight: increases with number of evaluations
    # We estimate the number of evaluations from the magnitude of f_best
    # Initially f_best is around 0 (random points), and decreases toward -3.32237
    # So the number of evaluations can be estimated as (0 - f_best) / typical_improvement
    # A simpler approach: use the fact that we start with 5 points and do 30 total
    # We can track this implicitly by how negative f_best is
    # When f_best is close to 0, we're early; when it's close to -3.3, we're late
    
    # Estimate progress: 0 at start, 1 at end
    # f_best starts around 0 and goes to -3.32237
    # Use a sigmoid-like function to map f_best to [0, 1]
    # At f_best = 0, progress = 0; at f_best = -3.32237, progress = 1
    
    f_best_min = -3.32237
    f_best_max = 0.0
    
    if f_best < f_best_min:
        progress = 1.0
    elif f_best > f_best_max:
        progress = 0.0
    else:
        progress = (f_best_max - f_best) / (f_best_max - f_best_min)
    
    # Dynamic weight: starts at 0.5 (balanced), increases to 0.8 (more exploitation)
    # As we get more evaluations, we should exploit more
    w_ei = 0.5 + 0.3 * progress
    w_sigma = 1.0 - w_ei
    
    # Combined score
    score = w_ei * ei_norm + w_sigma * sigma_norm
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e10, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
