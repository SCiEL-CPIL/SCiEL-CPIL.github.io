"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 300.0  # seconds, safety margin before 330s kill
_TIMEOUT_MARGIN = 25.0  # return at least 25s before deadline

def _check_timeout():
    return (time.time() - _START_TIME) > (_DEADLINE - _TIMEOUT_MARGIN)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization (Rastrigin 2D).
    
    Strategy: Hybrid Expected Improvement (EI) and Probability of Improvement (PoI)
    with adaptive exploration and cost-awareness.
    
    For multimodal functions like Rastrigin, pure EI can be overly conservative,
    while pure PoI lacks magnitude information. We blend them:
    - EI provides the expected magnitude of improvement.
    - PoI provides a robust probability signal for finding *any* improvement.
    - The blend weight adapts: more PoI early (exploration), more EI late (exploitation).
    
    Additionally, we apply a cost-aware scaling to prefer cheaper evaluations.
    """
    if _check_timeout():
        # Fallback: simple cost-aware exploitation
        mu = np.asarray(mu, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / cost_safe

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases: non-positive sigma
    sigma_safe = np.where(sigma > 1e-8, sigma, 1e-8)
    cost_safe = np.where(cost > 1e-8, cost, 1e-8)
    
    # Compute z-score
    diff = f_best - mu
    z = diff / sigma_safe
    z = np.clip(z, -10, 10)  # Clip for numerical stability
    
    # Expected Improvement (EI)
    ei = diff * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI should be non-negative
    
    # Probability of Improvement (PoI)
    poi = norm.cdf(z)
    
    # Adaptive blend:
    # Use mean cost as a proxy for "budget spent".
    # Early (low cost): favor PoI (exploration, robust to GP miscalibration).
    # Late (high cost): favor EI (exploitation, magnitude of improvement).
    # Cost ranges from ~5 (start) to 30 (end).
    # Let's map cost to a weight for EI: w_ei = cost / 35
    # At cost=5: w_ei = 0.14 -> 14% EI, 86% PoI
    # At cost=30: w_ei = 0.86 -> 86% EI, 14% PoI
    # This ensures a smooth transition.
    
    mean_cost = np.mean(cost)
    w_ei = np.clip(mean_cost / 35.0, 0.1, 0.9)
    w_poi = 1.0 - w_ei
    
    # Normalize EI and PoI to comparable scales.
    # PoI is in [0, 1]. EI can be larger.
    # We can scale EI by a factor to make it comparable to PoI.
    # A simple heuristic: scale EI by 1/(sigma_scale + 1) or just use raw EI.
    # However, mixing raw EI and PoI directly might be dominated by EI if sigma is large.
    # Let's try: score = w_ei * ei + w_poi * (poi * max_ei)
    # Or better: score = w_ei * ei + w_poi * poi
    # Since EI ~ O(sigma) and PoI ~ O(1), if sigma is large, EI dominates.
    # This is actually desirable: if uncertainty is high, EI is large, which is good.
    # But we want PoI to help with exploration in regions where EI might be low but PoI is high?
    # Actually, if EI is low, it means expected improvement is low. PoI high means likely improvement.
    # For minimization, if mu is close to f_best, EI is small, PoI is ~0.5.
    # If mu is far, EI is large, PoI is ~1.
    # The blend should work reasonably well.
    
    # Let's refine:
    # score = w_ei * ei + w_poi * (poi * np.max(ei) if np.any(ei>0) else 0)
    # Using max(ei) makes the PoI term scale with the best EI found, which is adaptive.
    max_ei = np.max(ei)
    if max_ei < 1e-9:
        max_ei = 1.0
        
    blended_score = w_ei * ei + w_poi * (poi * max_ei)
    
    # Cost-aware scaling
    # We want to prefer lower cost.
    # score = blended_score / cost_safe
    # This is the standard cost-aware approach.
    
    score = blended_score / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
