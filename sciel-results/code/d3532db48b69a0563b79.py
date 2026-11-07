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
    
    Uses a hybrid of Probability of Improvement (POI) and Expected Improvement (EI)
    with cost-aware scaling. POI focuses on high-confidence improvement regions,
    while EI captures magnitude of expected gain. The blend is weighted by uncertainty
    level and normalized by cost to favor cheap, promising evaluations.
    
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
    
    # Safety deadline guard: leave margin before 330s wall-clock
    DEADLINE = 320.0  # seconds
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-9)
    
    # --- Component 1: Probability of Improvement (POI) ---
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # This focuses on regions with high confidence of beating f_best
    delta = f_best - mu
    z_poi = delta / sigma_safe
    z_poi_clipped = np.clip(z_poi, -50, 50)
    poi = _norm_cdf(z_poi_clipped)
    poi = np.maximum(poi, 0.0)
    
    # --- Component 2: Expected Improvement (EI) ---
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    z_ei = delta / sigma_safe
    z_ei_clipped = np.clip(z_ei, -50, 50)
    ei = delta * _norm_cdf(z_ei_clipped) + sigma_safe * _norm_pdf(z_ei_clipped)
    ei = np.maximum(ei, 0.0)
    
    # --- Component 3: Lower Confidence Bound (LCB) for exploration ---
    # LCB(x) = mu - kappa * sigma, where kappa is cost-aware
    # Cheap points get larger exploration bonus
    cost_min = np.min(cost_safe)
    cost_max = np.max(cost_safe)
    cost_range = cost_max - cost_min if cost_max > cost_min else 1.0
    cost_norm = (cost_safe - cost_min) / cost_range  # 0 = cheapest, 1 = most expensive
    
    # Inverse cost scaling for exploration
    cost_multiplier = 1.0 / (1.0 + cost_norm)  # 1.0 (cheapest) to 0.5 (most expensive)
    kappa_base = 1.5
    kappa = kappa_base * cost_multiplier
    lcb = mu - kappa * sigma_safe
    
    # Normalize components to [0, 1]
    poi_max = np.max(poi) if np.max(poi) > 0 else 1.0
    poi_norm = poi / poi_max
    
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ei_norm = ei / ei_max
    
    lcb_min = np.min(lcb)
    lcb_max = np.max(lcb)
    lcb_range = lcb_max - lcb_min if lcb_max > lcb_min else 1.0
    # Lower LCB is better, so flip for higher = better scoring
    lcb_norm = 1.0 - (lcb - lcb_min) / lcb_range
    
    # Adaptive blending based on uncertainty level
    sigma_max = np.max(sigma_safe)
    sigma_relative = sigma_safe / (sigma_max + 1e-8)
    
    # Weight POI heavily for exploitation, EI for magnitude, LCB for exploration
    # More LCB weight when uncertainty is high
    w_poi = 0.5 + 0.2 * (1.0 - sigma_relative)
    w_ei = 0.3 + 0.1 * (1.0 - sigma_relative)
    w_lcb = 0.2 + 0.4 * sigma_relative
    
    # Normalize weights
    w_sum = w_poi + w_ei + w_lcb
    w_poi /= w_sum
    w_ei /= w_sum
    w_lcb /= w_sum
    
    combined = w_poi * poi_norm + w_ei * ei_norm + w_lcb * lcb_norm
    
    # Cost-awareness: divide by cost with power < 1 to be less aggressive
    scores = combined / np.power(cost_safe, 0.7)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline check
    if time.time() - t0 > DEADLINE:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
