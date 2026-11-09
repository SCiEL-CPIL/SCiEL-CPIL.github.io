"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: combines Expected Improvement (EI) with a UCB-style
    exploration term, adaptively weighted based on the current uncertainty
    landscape. This helps escape local minima in the multimodal Michalewicz 2D
    function while still exploiting promising regions.
    """
    # Guard against zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement component
    z_ei = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    
    # UCB component: linear in uncertainty
    # beta adapts: higher when uncertainty is large relative to improvement potential
    mean_sigma = np.mean(sigma_safe)
    max_improvement = np.maximum(f_best - mu, 0).max() if len(mu) > 0 else 0.0
    
    # Adaptive beta: scale exploration with uncertainty level
    # When sigma is large relative to the best improvement, explore more
    if max_improvement > 1e-9:
        beta = 1.5 + 0.5 * (mean_sigma / max_improvement)
    else:
        beta = 2.0
    
    ucb = -mu + beta * sigma_safe
    
    # Normalize both components to [0, 1] range for stable combination
    ei_min, ei_max = ei.min(), ei.max()
    ei_range = ei_max - ei_min
    if ei_range > 1e-12:
        ei_norm = (ei - ei_min) / ei_range
    else:
        ei_norm = np.ones_like(ei) * 0.5
    
    ucb_min, ucb_max = ucb.min(), ucb.max()
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-12:
        ucb_norm = (ucb - ucb_min) / ucb_range
    else:
        ucb_norm = np.ones_like(ucb) * 0.5
    
    # Weighted combination: more exploration early (high sigma), more exploitation later
    # Use a dynamic weight based on the ratio of uncertainty to improvement
    if max_improvement > 1e-9 and mean_sigma > 1e-9:
        uncertainty_ratio = mean_sigma / max_improvement
        w_ucb = np.clip(0.3 + 0.4 * uncertainty_ratio, 0.3, 0.7)
    else:
        w_ucb = 0.5
    w_ei = 1.0 - w_ucb
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
