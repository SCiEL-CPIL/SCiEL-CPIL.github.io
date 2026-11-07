# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Beale 2D.
    
    Uses a hybrid approach:
    1. Expected Improvement (EI) for exploitation
    2. UCB-style uncertainty bonus for exploration
    3. Adaptive weighting that shifts from exploration to exploitation as f_best decreases
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points
    f_best : float
        Best (lowest) observed value so far
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores, higher means evaluate next
    """
    start = time.time()
    limit = 1.0  # 1 second limit per call, plenty for N=2000
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Handle edge cases where EI might be NaN or inf
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # UCB-style term: -mu + beta * sigma
    # Adaptive beta: higher exploration early (large f_best), more exploitation later
    # Beale function has global min 0 at (3, 0.5), local minima elsewhere
    # With 30 evaluations, need to find global basin quickly
    
    # Beta decays as f_best approaches 0
    # When f_best is large (early), beta is high (~5) for exploration
    # When f_best is small (late), beta is low (~1) for exploitation
    beta = 1.0 + 4.0 * (f_best / (f_best + 1.0))
    
    ucb = -mu + beta * sigma_safe
    ucb = np.nan_to_num(ucb, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Check if computation took too long
    if time.time() - start > limit:
        # Fallback to pure EI if something goes wrong
        return ei
    
    # Normalize both terms to comparable scales
    ei_max = np.max(ei)
    if ei_max > 1e-8:
        ei_norm = ei / ei_max
    else:
        ei_norm = ei
    
    ucb_max = np.max(ucb)
    ucb_min = np.min(ucb)
    if ucb_max - ucb_min > 1e-8:
        ucb_norm = (ucb - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb)
    
    # Adaptive weight: shift from UCB (exploration) to EI (exploitation)
    # When f_best is large, weight more on UCB; when small, weight more on EI
    # progress = 0 means f_best is very large (early), progress = 1 means f_best ~ 0 (late)
    progress = 1.0 - (f_best / (f_best + 1.0))
    
    # Weight for EI increases with progress
    # w_ei goes from ~0.2 (early) to ~0.8 (late)
    w_ei = 0.2 + 0.6 * progress
    w_ucb = 1.0 - w_ei
    
    # Combine
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
