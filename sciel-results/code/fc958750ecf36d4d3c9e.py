import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of styblinski tang 1d.
    
    Uses a hybrid of Expected Improvement (EI) and Upper Confidence Bound (UCB).
    EI handles exploitation while UCB provides structured exploration.
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Calculate z-score for EI
    z = (f_best - mu) / sigma
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB: balance exploitation (-mu) and exploration (beta * sigma)
    # beta controls the exploration-exploitation tradeoff
    beta = 1.0
    ucb = -mu + beta * sigma
    
    # Normalize both to [0, 1] range for blending
    ei_min, ei_max = np.min(ei), np.max(ei)
    ucb_min, ucb_max = np.min(ucb), np.max(ucb)
    
    # Avoid division by zero
    ei_range = ei_max - ei_min + 1e-9
    ucb_range = ucb_max - ucb_min + 1e-9
    
    ei_norm = (ei - ei_min) / ei_range
    ucb_norm = (ucb - ucb_min) / ucb_range
    
    # Blend: weight EI more as we get closer to optimum (lower f_best relative to mu spread)
    # Use a fixed blend ratio that favors EI slightly
    weight_ei = 0.6
    weight_ucb = 0.4
    
    score = weight_ei * ei_norm + weight_ucb * ucb_norm
    
    # Scale back to original EI scale for consistency
    score = score * ei_range + ei_min
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Time guard
    if time.time() - start_time > 290:
        return -mu + 0.1 * sigma

    return score
