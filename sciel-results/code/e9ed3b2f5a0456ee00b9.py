"""Combined cost-aware acquisition: PoI-based with adaptive exploration and diversity."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 310.0  # 20s safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Probability of Improvement (PoI)
    with adaptive exploration weighting and diversity promotion via GP variance.
    
    The score combines:
    1. PoI: likelihood of improving over f_best (bounded [0,1])
    2. Adaptive exploration: sigma-weighted term that adapts based on f_best
    3. Diversity: penalizes points where GP uncertainty is low (already explored)
    
    Final score = (PoI + alpha * sigma / cost_scale) * diversity_weight / cost
    
    Parameters:
    -----------
    mu : array-like, shape (n,)
        GP mean predictions
    sigma : array-like, shape (n,)
        GP standard deviation predictions
    f_best : float
        Best (lowest) observed value so far
    cost : array-like, shape (n,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition values (higher = more desirable to evaluate)
    """
    # Deadline guard: return simple fallback to avoid timeout
    if time.time() > _DEADLINE:
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-9)
        score = sigma_safe / (cost + 1e-9)
        return np.where(np.isfinite(score), score, 0.0)

    # Convert to numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 1e-9)

    # Compute z-score for Probability of Improvement
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -50, 50)
    
    # PoI = Phi(z), bounded in [0, 1]
    poi = norm.cdf(z)

    # Adaptive exploration weight:
    # When f_best is close to 0 (good), reduce exploration
    # When f_best is far from 0, increase exploration
    # f_best for griewank is in [0, ~1], typically starts around 0.5-1.0
    # Normalize f_best: f_best=0 means we're done, f_best=1 means early
    f_best_norm = np.clip(f_best / 1.0, 0.0, 1.0)
    
    # Alpha interpolates from high exploration (early) to low exploration (late)
    # Early: alpha ~ 2.0, Late: alpha ~ 0.5
    alpha = 2.0 - 1.5 * (1.0 - f_best_norm)
    
    # Exploration term: alpha * sigma, normalized to be comparable to PoI
    # sigma is in objective units; for griewank, typical sigma ~ 0.01-0.1
    # Scale sigma to be in similar range as PoI
    sigma_scale = np.max(sigma_safe) if np.max(sigma_safe) > 1e-10 else 1.0
    exploration_term = alpha * (sigma_safe / (sigma_scale + 1e-10))
    
    # Combine PoI and exploration
    base_score = poi + exploration_term

    # Diversity promotion: use GP variance as proxy for information gain
    # Points with high sigma are in unexplored regions -> higher diversity weight
    # Points with low sigma are in well-explored regions -> lower diversity weight
    # Use a soft normalization: diversity_weight = sigma / (sigma + sigma_median)
    sigma_median = np.median(sigma_safe)
    if sigma_median < 1e-10:
        sigma_median = 1e-10
    
    diversity_weight = sigma_safe / (sigma_safe + sigma_median)
    # Clip to [0.3, 1.0] to avoid zeroing out any point entirely
    diversity_weight = np.clip(diversity_weight, 0.3, 1.0)

    # Cost normalization: divide by cost to prefer cheaper points
    # Cost is in [1, 2] for this task
    cost_safe = np.maximum(cost, 1e-9)
    
    # Final score
    score = base_score * diversity_weight / cost_safe

    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
