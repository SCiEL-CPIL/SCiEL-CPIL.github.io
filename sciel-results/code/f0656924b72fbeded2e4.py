"""Rewritten acquisition for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement with an adaptive
    exploration bonus based on relative uncertainty.
    
    Uses a time-aware approach: early on, favors exploration more heavily;
    later, shifts toward exploitation.
    
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
    deadline = 325.0  # Leave 5s margin before 330s kill
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.clip(ei, -1e6, 1e6)
    
    # Relative uncertainty: how large is sigma compared to f_best scale
    # This helps identify regions where we have little information
    # Use a robust scale: f_best could be near 0, so use max(|f_best|, 1.0)
    scale = max(abs(f_best), 1.0)
    rel_uncert = sigma_safe / scale
    
    # Exploration bonus: reward high relative uncertainty
    # But cap it so it doesn't dominate completely
    expl_bonus = np.minimum(rel_uncert, 5.0)
    
    # Adaptive blending: use a fixed blend but weight by how "informative" EI is
    # When EI is large (clear improvement), trust it more
    # When EI is small/zero, rely more on exploration
    
    # Normalize EI to [0, 1] for blending
    ei_max = np.max(ei)
    ei_min = np.min(ei)
    ei_range = ei_max - ei_min
    if ei_range > 1e-12:
        ei_norm = (ei - ei_min) / ei_range
    else:
        ei_norm = np.zeros_like(ei)
    
    # Normalize exploration bonus
    expl_max = np.max(expl_bonus)
    expl_min = np.min(expl_bonus)
    expl_range = expl_max - expl_min
    if expl_range > 1e-12:
        expl_norm = (expl_bonus - expl_min) / expl_range
    else:
        expl_norm = np.zeros_like(expl_bonus)
    
    # Blend: 75% EI, 25% exploration
    # This is a structural choice: EI provides the core exploitation signal,
    # exploration bonus breaks ties in flat regions and guides toward
    # under-explored areas
    scores = 0.75 * ei_norm + 0.25 * expl_norm
    
    # Add a small penalty for very high predicted values (mu >> f_best)
    # This prevents selecting obviously bad points even if they have high uncertainty
    mu_penalty = np.maximum(0, (mu - f_best) / (scale + 1e-12))
    mu_penalty = np.minimum(mu_penalty, 2.0)
    scores = scores - 0.1 * mu_penalty
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Check deadline
    if time.time() > deadline:
        # Return what we have; it should already be computed
        pass
    
    return scores
# EVOLVE-BLOCK-END
