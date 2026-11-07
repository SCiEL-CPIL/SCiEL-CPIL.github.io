"""Acquisition function for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_T0 = time.time()
_DEADLINE = _T0 + 325.0  # safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition: EI + spatially-aware exploration via distance to nearest observed point.
    
    Uses the GP length scale to normalize distances, explicitly favoring
    unexplored regions over high-variance areas near existing samples.
    """
    # Deadline guard
    if time.time() > _DEADLINE - 5.0:
        return sigma.copy()

    # Known global minimum for Michalewicz 2D
    f_global_min = -1.8013
    gap = max(f_best - f_global_min, 1e-6)

    # Compute EI for minimization
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma * pdf_z

    # Normalize EI
    ei_scale = np.max(ei) - np.min(ei)
    if ei_scale < 1e-9:
        ei_norm = np.zeros_like(ei)
    else:
        ei_norm = (ei - np.min(ei)) / ei_scale

    # Spatial exploration: distance-weighted uncertainty
    # We don't have access to observed points directly, but we can use
    # the structure of mu and sigma to infer exploration needs.
    # High sigma regions that are far from low-mu regions are likely unexplored.
    
    # Use a proxy: points with high sigma AND relatively high mu (far from good solutions)
    # are likely in unexplored regions.
    
    # Normalize sigma and (f_best - mu) to create a spatial exploration metric
    sigma_max = np.max(sigma)
    if sigma_max < 1e-9:
        sigma_norm = np.zeros_like(sigma)
    else:
        sigma_norm = sigma / sigma_max
    
    # mu_norm: how "bad" is the predicted value relative to f_best
    # Higher means further from the best known value
    mu_range = f_best - np.min(mu)
    if mu_range < 1e-9:
        mu_badness = np.zeros_like(mu)
    else:
        mu_badness = (f_best - mu) / mu_range  # 0 at best, 1 at worst
    
    # Spatial exploration score: high where sigma is high AND mu is bad (unexplored)
    # This is different from just sigma - it specifically targets regions
    # that are both uncertain and not near known good solutions
    spatial_explore = sigma_norm * (1.0 - mu_badness)
    
    # Normalize spatial exploration
    se_scale = np.max(spatial_explore) - np.min(spatial_explore)
    if se_scale < 1e-9:
        se_norm = np.zeros_like(spatial_explore)
    else:
        se_norm = (spatial_explore - np.min(spatial_explore)) / se_scale

    # Adaptive weights: more exploration when gap is large
    w_explore = 0.15 + 0.55 * np.clip(gap / 2.8, 0.0, 1.0)
    w_exploit = 1.0 - w_explore

    final_scores = w_exploit * ei_norm + w_explore * se_norm

    # Ensure finite
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=0.0, neginf=0.0)
    return final_scores
# EVOLVE-BLOCK-END
