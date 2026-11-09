"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 3D.
    Uses a hybrid of Expected Improvement (EI) and Lower Confidence Bound (LCB)
    with adaptive weighting based on uncertainty and progress.
    """
    # Guard against zero/negative sigma
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute LCB score (we want to maximize, so we use -LCB = -mu + beta*sigma)
    # Use an adaptive beta that decreases as we get more confident
    # Start with higher exploration, reduce over time
    # We can use the number of unique finite mu values as a proxy for progress
    # Or just use a fixed moderate beta since we don't have step count
    beta = 1.5
    lcb_score = -mu + beta * sigma_safe
    
    # Normalize both to [0, 1] range for comparable weighting
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min + 1e-9
    ei_norm = (ei - ei_min) / ei_range
    
    lcb_min = np.min(lcb_score)
    lcb_max = np.max(lcb_score)
    lcb_range = lcb_max - lcb_min + 1e-9
    lcb_norm = (lcb_score - lcb_min) / lcb_range
    
    # Adaptive weighting: give more weight to EI when uncertainty is low
    # and more weight to LCB when uncertainty is high
    # Use the mean normalized sigma as a proxy for exploration needs
    sigma_mean = np.mean(sigma_safe)
    sigma_min = np.min(sigma_safe)
    sigma_max = np.max(sigma_safe)
    sigma_range = sigma_max - sigma_min + 1e-9
    sigma_norm_mean = (sigma_mean - sigma_min) / sigma_range
    
    # Weight between 0.3 (more exploration) and 0.7 (more exploitation)
    # When sigma is high, we want more exploration (lower EI weight)
    # When sigma is low, we want more exploitation (higher EI weight)
    ei_weight = 0.7 - 0.4 * sigma_norm_mean
    lcb_weight = 1.0 - ei_weight
    
    scores = ei_weight * ei_norm + lcb_weight * lcb_norm
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
