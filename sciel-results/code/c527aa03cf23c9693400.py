import numpy as np
import time
from scipy.stats import norm

# Global deadline tracking
_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    """
    Adaptive cost-aware acquisition function combining UCB and Expected Improvement.
    
    Uses a hybrid approach that adapts based on f_best:
    - When f_best is large (early phase): use UCB for exploration
    - When f_best is small (late phase): use EI for exploitation
    - Cost-awareness is strengthened near the optimum to conserve budget
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions.
    sigma : ndarray, shape (2000,)
        GP standard deviations.
    f_best : float
        Best (lowest) observed objective value.
    cost : ndarray, shape (2000,)
        Evaluation costs for each candidate.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    global _start_time, _call_count
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 325.0:
        mu_arr = np.asarray(mu)
        # Fallback: pure exploitation
        fallback_scores = -mu_arr
        return np.where(np.isfinite(fallback_scores), fallback_scores, 0.0)
    
    _call_count += 1
    
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Safety clamps
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Adaptive parameters based on f_best and progress
    # f_best for three-hump camel 2d ranges from ~0 to ~10
    # When f_best is small, we're close to optimum -> exploit more
    f_best_val = float(f_best)
    
    # Progress-based decay
    max_steps = 20.0
    progress = min(_call_count / max_steps, 1.0)
    
    # f_best-based adaptation: stronger exploitation when f_best is small
    # Normalize f_best: 0 means at optimum, 10 means far away
    f_best_norm = np.clip(f_best_val / 10.0, 0.0, 1.0)
    
    # Combined progress: both time and f_best matter
    # When f_best is small, we should exploit more aggressively
    effective_progress = 0.5 * progress + 0.5 * (1.0 - f_best_norm)
    
    # Adaptive kappa for UCB component
    # Higher kappa when f_best is large (exploration), lower when f_best is small (exploitation)
    kappa_max = 3.0
    kappa_min = 0.3
    kappa = kappa_max - (kappa_max - kappa_min) * effective_progress
    
    # UCB for minimization: Score = -mu + kappa * sigma
    ucb_scores = -mu + kappa * sigma_safe
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    z = (f_best_val - mu) / sigma_safe
    z = np.clip(z, -10.0, 10.0)  # Clip for numerical stability
    ei_scores = (f_best_val - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei_scores = np.maximum(ei_scores, 0.0)  # EI should be non-negative
    
    # Blend UCB and EI based on f_best
    # When f_best is small, weight more toward EI (exploitation)
    # When f_best is large, weight more toward UCB (exploration)
    ei_weight = 0.3 + 0.7 * (1.0 - f_best_norm)  # 0.3 to 1.0
    ucb_weight = 1.0 - ei_weight
    
    # Normalize scores for blending
    ucb_mean = np.mean(np.abs(ucb_scores))
    ei_mean = np.mean(np.abs(ei_scores))
    
    # Scale to comparable ranges
    ucb_scaled = ucb_scores / (ucb_mean + 1e-8)
    ei_scaled = ei_scores / (ei_mean + 1e-8)
    
    combined_scores = ucb_weight * ucb_scaled + ei_weight * ei_scaled
    
    # Cost-awareness: divide by cost, but strengthen near optimum
    # When f_best is small, be more conservative about expensive evaluations
    cost_factor = 1.0 + 0.5 * (1.0 - f_best_norm)  # 1.0 to 1.5
    scores = combined_scores / (cost_safe ** cost_factor)
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
