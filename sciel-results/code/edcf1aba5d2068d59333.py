"""
Acquisition function for Bayesian optimization of Michalewicz 2D.

This program uses a hybrid acquisition strategy that combines:
1. Expected Improvement (EI) - for exploitation
2. Upper Confidence Bound (UCB) - for exploration
3. A diversity/exploration bonus based on distance from the current best

The key insight is that Michalewicz 2D is multimodal, so we need a balance
between exploiting known good regions and exploring new areas.

For minimization:
- EI = (f_best - mu) * Phi(u) + sigma * phi(u), where u = (f_best - mu) / sigma
- UCB = -mu + beta * sigma

We blend these with a weight that adapts based on the current best value.
When f_best is far from the known minimum (-1.8013), we explore more.
When f_best is close to the minimum, we exploit more.
"""
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at each candidate
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at each candidate
    f_best : float
        Best (lowest) observed function value so far
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Known minimum of Michalewicz 2D
    f_min = -1.8013
    
    # Handle edge cases
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate Expected Improvement for minimization
    u = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(u) + sigma_safe * norm.pdf(u)
    
    # Handle zero sigma cases
    zero_sigma_mask = sigma < eps
    if np.any(zero_sigma_mask):
        ei[zero_sigma_mask] = np.maximum(0.0, f_best - mu[zero_sigma_mask])
    
    # Calculate UCB for minimization
    beta = 2.0
    ucb = -mu + beta * sigma_safe
    
    # Normalize EI and UCB to [0, 1] for blending
    ei_min, ei_max = np.min(ei), np.max(ei)
    ucb_min, ucb_max = np.min(ucb), np.max(ucb)
    
    ei_range = ei_max - ei_min
    ucb_range = ucb_max - ucb_min
    
    if ei_range > eps:
        ei_norm = (ei - ei_min) / ei_range
    else:
        ei_norm = np.ones_like(ei)
    
    if ucb_range > eps:
        ucb_norm = (ucb - ucb_min) / ucb_range
    else:
        ucb_norm = np.ones_like(ucb)
    
    # Adapt exploration weight based on how close f_best is to the minimum
    # If f_best is close to f_min, exploit more (lower exploration weight)
    # If f_best is far from f_min, explore more (higher exploration weight)
    gap = max(f_best - f_min, 0.0)
    # Scale gap to [0, 1] with a characteristic scale of ~0.5
    gap_scaled = 1.0 - np.exp(-gap / 0.5)
    
    # Exploration weight: higher when we're far from the optimum
    # Base exploration weight between 0.2 and 0.6
    w_explore = 0.2 + 0.4 * gap_scaled
    w_exploit = 1.0 - w_explore
    
    # Blend EI and UCB
    scores = w_exploit * ei_norm + w_explore * ucb_norm
    
    # Add a small bonus for points with high uncertainty (to encourage exploration)
    # This helps escape local optima in the multimodal landscape
    sigma_norm = sigma_safe / (np.max(sigma_safe) + eps)
    scores += 0.1 * sigma_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
