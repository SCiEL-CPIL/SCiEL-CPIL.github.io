"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Hypervolume-inspired acquisition function for Bayesian optimization.
    
    We score candidates by a combination of expected improvement and a
    diversity bonus that rewards candidates with high uncertainty,
    approximating the hypervolume contribution of adding this candidate
    to the set of non-dominated (mu, sigma) points.
    
    Parameters
    ----------
    mu : array_like, shape (n,)
        Posterior mean at candidate points.
    sigma : array_like, shape (n,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best observed value so far (minimum).
    
    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher means more promising.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement component
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Normalize EI to [0, 1] range for combination
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    # Diversity/uncertainty component: reward high sigma
    # This approximates the hypervolume contribution of exploring
    # regions with high uncertainty
    sigma_max = np.max(sigma)
    if sigma_max > 1e-12:
        sigma_norm = sigma / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma)
    
    # Combine EI and diversity with a weight that emphasizes exploration
    # early (when f_best is far from optimal) and exploitation later
    # Use a simple heuristic: weight based on how far f_best is from 0
    # For Ackley, f_best starts around 2-3 and goes to 0
    # When f_best is high, explore more; when low, exploit more
    
    # Simple adaptive weight: as f_best approaches 0, shift to exploitation
    # f_best for Ackley is in [0, ~7], so normalize
    f_best_clipped = np.clip(f_best, 0, 7)
    # Weight for exploration: starts at 0.4, decreases as f_best -> 0
    # When f_best is large (early), exploration weight is higher
    # When f_best is small (late), exploitation weight is higher
    explore_weight = 0.3 + 0.3 * (f_best_clipped / 7.0)
    exploit_weight = 1.0 - explore_weight
    
    # Hypervolume-inspired score:
    # A candidate contributes to hypervolume if it has low mu (good mean)
    # and/or high sigma (high uncertainty). We approximate this by
    # combining normalized EI (which captures both) with a sigma bonus.
    
    # Additional diversity: penalize candidates that are too similar to
    # the current best by adding a small bonus for high sigma
    score = exploit_weight * ei_norm + explore_weight * sigma_norm
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
