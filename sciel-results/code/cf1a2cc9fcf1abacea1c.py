"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import erf


def _norm_cdf(x):
    """Vectorized standard normal CDF using erf."""
    x = np.asarray(x, dtype=np.float64)
    result = np.empty_like(x)
    
    mask_large_pos = x > 8.0
    mask_large_neg = x < -8.0
    mask_mid = ~(mask_large_pos | mask_large_neg)
    
    result[mask_large_pos] = 1.0
    result[mask_large_neg] = 0.0
    
    if np.any(mask_mid):
        xm = x[mask_mid]
        result[mask_mid] = 0.5 * (1.0 + erf(xm / np.sqrt(2.0)))
    
    return result


def _norm_pdf(x):
    """Vectorized standard normal PDF."""
    x = np.asarray(x, dtype=np.float64)
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for minimizing shekel 4d.
    
    Combines the cost-aware LCB structure of the current program with
    a probability-of-significant-improvement (POI) component that targets
    high-confidence gains over a fixed threshold, blended adaptively.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions (in objective units).
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions (in objective units).
    f_best : float
        Lowest observed function value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    t0 = time.time()
    
    # Safety deadline guard
    DEADLINE = 320.0
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-9)
    
    # --- Component 1: Cost-aware LCB (from current program) ---
    # Normalized cost in [0, 1]
    cost_min = np.min(cost_safe)
    cost_max = np.max(cost_safe)
    cost_range = cost_max - cost_min if cost_max > cost_min else 1.0
    cost_norm = (cost_safe - cost_min) / cost_range
    
    # Inverse cost scaling: cheap points get higher exploration multiplier
    cost_multiplier = 1.0 / (1.0 + cost_norm)  # 1.0 (cheapest) to 0.5 (most expensive)
    
    # LCB with dynamic kappa
    kappa_base = 1.0
    kappa = kappa_base * cost_multiplier
    lcb = mu - kappa * sigma_safe
    
    # LCB score: lower LCB is better, so negate
    lcb_score = -lcb
    
    # --- Component 2: Probability of Significant Improvement (POI) ---
    # Target a meaningful improvement threshold
    # Scale epsilon relative to the magnitude of f_best
    typical_scale = max(abs(f_best), 0.1)
    eps = 0.05 * typical_scale
    target = f_best - eps
    
    z_poi = (target - mu) / sigma_safe
    z_poi_clipped = np.clip(z_poi, -50, 50)
    poi = _norm_cdf(z_poi_clipped)
    poi = np.maximum(poi, 0.0)
    
    # Normalize POI to [0, 1]
    poi_max = np.max(poi) if np.max(poi) > 0 else 1.0
    poi_norm = poi / poi_max
    
    # --- Component 3: Expected Improvement (EI) for magnitude ---
    delta = f_best - mu
    z_ei = delta / sigma_safe
    z_ei_clipped = np.clip(z_ei, -50, 50)
    ei = delta * _norm_cdf(z_ei_clipped) + sigma_safe * _norm_pdf(z_ei_clipped)
    ei = np.maximum(ei, 0.0)
    
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ei_norm = ei / ei_max
    
    # --- Adaptive blending ---
    # When uncertainty is high, weight LCB (exploration) more
    # When uncertainty is low, weight POI and EI (exploitation) more
    sigma_max = np.max(sigma_safe)
    sigma_relative = sigma_safe / (sigma_max + 1e-8)
    
    # Weights: POI for confidence, EI for magnitude, LCB for exploration
    w_poi = 0.4 + 0.2 * (1.0 - sigma_relative)
    w_ei = 0.2 + 0.1 * (1.0 - sigma_relative)
    w_lcb = 0.4 + 0.3 * sigma_relative
    
    w_sum = w_poi + w_ei + w_lcb
    w_poi /= w_sum
    w_ei /= w_sum
    w_lcb /= w_sum
    
    # Normalize LCB score to [0, 1]
    lcb_min = np.min(lcb_score)
    lcb_max = np.max(lcb_score)
    lcb_range = lcb_max - lcb_min if lcb_max > lcb_min else 1.0
    lcb_norm = (lcb_score - lcb_min) / lcb_range
    
    # Combined score before cost normalization
    combined = w_poi * poi_norm + w_ei * ei_norm + w_lcb * lcb_norm
    
    # Cost-awareness: divide by cost with a power to penalize expensive points
    # Using power 0.7 for moderate cost sensitivity
    scores = combined / np.power(cost_safe, 0.7)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline check
    if time.time() - t0 > DEADLINE:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
