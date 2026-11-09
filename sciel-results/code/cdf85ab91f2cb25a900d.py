import numpy as np
from scipy.stats import norm
import time

# Global time tracking for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using adaptive EI/PoI blend
    for Bayesian optimization on Shekel 4d.
    """
    global _start_time
    
    # Initialize time tracking
    if _start_time is None:
        _start_time = time.time()
    
    # Ensure arrays are 1D and finite
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    cost = np.asarray(cost, dtype=np.float64).ravel()
    
    # Safety: replace non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Deadline guard: if we've used most of the 330s budget, return quickly
    elapsed = time.time() - _start_time
    if elapsed > 320:  # 10s safety margin
        # Return a simple, fast score to avoid timeout
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        poi = norm.cdf(z)
        score = poi / cost
        score = np.where(np.isfinite(score), score, 0.0)
        return score
    
    # Expected Improvement
    z = (f_best - mu) / (sigma + 1e-12)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Adaptive PoI/EI mix
    # PoI (Probability of Improvement) is more exploration-friendly than EI
    poi = norm.cdf(z)
    
    # Blend EI and PoI using a smooth, data-driven weight
    # Use the ratio of local sigma to global sigma scale
    sigma_median = np.median(sigma) if len(sigma) > 0 else 1.0
    sigma_scale = np.maximum(sigma_median, 1e-6)
    
    # Smooth blend: when sigma is high relative to scale, favor exploration (PoI)
    # Use a smoother logistic-like transition
    blend_weight = sigma / (sigma + 0.3 * sigma_scale)
    # Clip to [0, 1] for numerical safety
    blend_weight = np.clip(blend_weight, 0.0, 1.0)
    
    # Combined improvement metric
    improvement = blend_weight * poi + (1 - blend_weight) * ei
    
    # Reduced exploration bonus to avoid over-exploration
    # Scale by a smaller factor since the blend already handles exploration
    exploration_bonus = 0.1 * sigma
    
    # Total acquisition score
    total_score = improvement + exploration_bonus
    
    # Cost-aware normalization
    # Cost is in [1, 2] range, so dividing by cost gives ~0.5-1.0 weighting
    score = total_score / cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
