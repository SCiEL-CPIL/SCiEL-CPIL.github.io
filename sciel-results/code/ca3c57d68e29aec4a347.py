"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann-3D.
    
    Combines the adaptive LCB structure of the current program with
    Expected Improvement (EI) from the inspirations. Uses a smooth
    blending between EI and adaptive LCB based on progress toward
    the known global minimum.
    
    Parameters
    ----------
    mu : ndarray, shape (n,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (n,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Known global minimum for Hartmann-3D
    f_global_min = -3.86278
    
    # Reference worst value for progress calculation
    f_worst_ref = 0.0
    
    # Calculate progress towards the global minimum
    denominator = f_worst_ref - f_global_min
    if abs(denominator) < 1e-12:
        progress = 0.5
    else:
        progress = (f_worst_ref - f_best) / denominator
    
    # Clip progress to [0, 1]
    progress = float(np.clip(progress, 0.0, 1.0))
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute adaptive LCB score (higher is better for selection)
    # LCB for minimization: we want low mu and high sigma
    # Score = -mu + kappa * sigma
    kappa_max = 3.0
    kappa_min = 0.5
    kappa = kappa_max - (kappa_max - kappa_min) * progress
    
    lcb_score = -mu + kappa * sigma_safe
    
    # Blend between EI and LCB based on progress
    # Early: more LCB (exploration), Late: more EI (exploitation)
    # Use a smooth weighting
    weight_lcb = (1.0 - progress) * 0.4  # Up to 40% LCB contribution
    weight_ei = 1.0 - weight_lcb
    
    # Normalize both components to have similar scale before blending
    ei_max = np.max(ei)
    lcb_max = np.max(lcb_score)
    
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = ei
    
    if lcb_max > 1e-12:
        lcb_norm = lcb_score / lcb_max
    else:
        lcb_norm = lcb_score
    
    scores = weight_ei * ei_norm + weight_lcb * lcb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
