"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Thompson Sampling proxy: maximize (f_best - mu) / (sigma + epsilon)
    # This encourages points with high expected improvement relative to uncertainty
    # Also add a small EI component for stability
    
    epsilon = 1e-6
    ts_score = (f_best - mu) / (sigma + epsilon)
    
    # Compute EI for a small stabilizing component
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Normalize both to [0, 1]
    ts_min = np.min(ts_score)
    ts_max = np.max(ts_score)
    ts_range = ts_max - ts_min + 1e-9
    ts_norm = (ts_score - ts_min) / ts_range
    
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min + 1e-9
    ei_norm = (ei - ei_min) / ei_range
    
    # Weight TS more heavily (80-20)
    scores = 0.8 * ts_norm + 0.2 * ei_norm
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
