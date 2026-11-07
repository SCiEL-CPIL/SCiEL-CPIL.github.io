"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    
    Uses a hybrid approach combining:
    1. Lower Confidence Bound (LCB) for exploitation of promising regions
    2. Pure exploration bonus for regions with high uncertainty
    3. Adaptive beta based on progress (f_best relative to known minimum)
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Mean of the GP posterior at candidate points
    sigma : np.ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points
    f_best : float
        Best (lowest) observed function value so far
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores; higher means more likely to be evaluated next
    """
    # Track time for safety
    t_start = time.time()
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-10)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-10)
    mu_safe = np.where(np.isfinite(mu), mu, 0.0)
    
    # Known minimum of Michalewicz 2D
    f_min = -1.8013
    
    # Compute progress: how close are we to the global minimum?
    # If f_best is close to f_min, we should exploit more
    # If f_best is far, we should explore more
    if f_best is not None and np.isfinite(f_best):
        progress = np.clip((0 - f_best) / (0 - f_min), 0, 1) if f_min < 0 else 0
    else:
        progress = 0
    
    # Adaptive beta: start high for exploration, decrease as we make progress
    # At the beginning (progress=0), beta is high to explore
    # As we get closer to the minimum, beta decreases to exploit
    beta = 2.5 * (1 - 0.5 * progress) + 0.5
    
    # LCB score for minimization: we want to minimize mu - beta*sigma
    # So the acquisition score (higher is better) is: -(mu - beta*sigma) = -mu + beta*sigma
    lcb_score = -mu_safe + beta * sigma_safe
    
    # Add an exploration bonus: reward regions where sigma is high relative to the range
    # This helps escape local optima in the multimodal landscape
    sigma_max = np.max(sigma_safe) if len(sigma_safe) > 0 else 1.0
    sigma_min = np.min(sigma_safe) if len(sigma_safe) > 0 else 0.0
    sigma_range = sigma_max - sigma_min
    if sigma_range < 1e-10:
        exploration_bonus = 0.0
    else:
        # Normalize sigma to [0, 1] and use as exploration bonus
        sigma_norm = (sigma_safe - sigma_min) / sigma_range
        exploration_bonus = 0.3 * sigma_norm
    
    # Combine: LCB score plus exploration bonus
    scores = lcb_score + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Safety check: if we're taking too long, return early
    if time.time() - t_start > 5.0:
        # Fallback: just use pure exploitation
        scores = -mu_safe
    
    return scores
# EVOLVE-BLOCK-END
