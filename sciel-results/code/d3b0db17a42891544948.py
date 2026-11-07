import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 320.0

# State to track progress
_call_count = 0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    Uses a hybrid of Expected Improvement (EI) and Upper Confidence Bound (UCB)
    with adaptive exploration based on progress.
    
    Higher score means evaluate next.
    """
    global _call_count
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Return a safe fallback: prioritize low mu (exploitation)
        return -mu.copy()
    
    _call_count += 1
    
    # Determine exploration parameter based on progress
    n_steps = _call_count
    
    # Adaptive kappa: higher when early, lower when late
    if n_steps <= 5:
        base_kappa = 2.5  # Early: high exploration
    elif n_steps <= 15:
        base_kappa = 1.2  # Mid: moderate exploration
    else:
        base_kappa = 0.6  # Late: more exploitation
    
    # Also scale kappa with f_best to encourage larger improvements early
    # When f_best is large, we can afford to explore more
    f_best_scale = min(1.0, f_best / 10.0)
    kappa = base_kappa * (0.5 + 0.5 * f_best_scale)
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Compute EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    EI = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Compute UCB = -mu + kappa * sigma (for minimization, we want low mu + high uncertainty)
    UCB = -mu + kappa * sigma_safe
    
    # Weight them: more UCB early, more EI late
    if n_steps <= 8:
        ei_weight = 0.2
        ucb_weight = 0.8
    elif n_steps <= 18:
        ei_weight = 0.5
        ucb_weight = 0.5
    else:
        ei_weight = 0.8
        ucb_weight = 0.2
    
    # Normalize EI and UCB to similar scales for combination
    ei_max = np.max(EI) if np.max(EI) > 1e-10 else 1.0
    EI_normalized = EI / ei_max
    
    ucb_min = np.min(UCB)
    ucb_max = np.max(UCB)
    ucb_range = ucb_max - ucb_min
    if ucb_range < 1e-10:
        ucb_range = 1.0
    UCB_normalized = (UCB - ucb_min) / ucb_range
    
    # Combine
    score = ei_weight * EI_normalized + ucb_weight * UCB_normalized
    
    # Handle numerical issues
    score = np.nan_to_num(score, nan=0.0, posinf=1e10, neginf=0.0)
    
    return score
