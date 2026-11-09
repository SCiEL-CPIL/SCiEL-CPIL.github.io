"""Rewritten acquisition for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement with Student-t-like
    uncertainty weighting for better exploration in high-curvature regions.
    
    The key structural change: instead of a fixed blend of normalized EI and
    exploration, we use a modified EI that accounts for tail behavior.
    Points with high sigma relative to the scale of improvement get a boosted
    EI score, mimicking heavier-tailed posterior uncertainty.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points.
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising.
    """
    # Deadline guard
    t0 = time.time()
    deadline = 325.0
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Base Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.clip(ei, -1e6, 1e6)
    
    # Scale reference: use a robust estimate of the function scale
    # Beale function has minimum 0 and can reach large values
    # Use max(|f_best|, 1.0) as a reasonable scale
    scale = max(abs(f_best), 1.0)
    
    # Relative uncertainty
    rel_sigma = sigma_safe / scale
    
    # Student-t-like boost: for high uncertainty points, amplify the EI
    # The idea: if sigma is large relative to scale, the true improvement
    # could be much larger than the Gaussian EI suggests
    # Use a soft saturation to avoid extreme values
    # boost_factor = 1 + alpha * (rel_sigma / (1 + rel_sigma))
    # This gives a boost up to (1+alpha) for very high uncertainty
    alpha = 0.5
    boost = 1.0 + alpha * (rel_sigma / (1.0 + rel_sigma))
    
    # Apply boost to EI, but only when EI is positive (potential improvement)
    ei_boosted = np.where(ei > 0, ei * boost, ei)
    
    # Also add a direct uncertainty reward for exploration
    # This ensures we don't completely ignore high-uncertainty regions
    # even if their EI is currently low
    # Use a sigmoid-like function to smoothly transition
    expl_weight = 0.15
    exploration = expl_weight * (rel_sigma / (1.0 + rel_sigma))
    
    # Combine: boosted EI + exploration bonus
    scores = ei_boosted + exploration
    
    # Penalize points with very high predicted values
    # If mu is much larger than f_best, it's unlikely to be an improvement
    mu_excess = np.maximum(0, (mu - f_best) / scale)
    penalty = 0.1 * np.minimum(mu_excess, 3.0)
    scores = scores - penalty
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Check deadline
    if time.time() > deadline:
        pass
    
    return scores
# EVOLVE-BLOCK-END
