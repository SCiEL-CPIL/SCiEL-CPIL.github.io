import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 320s to be safe within 330s

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses an adaptive cost exponent to balance exploration and exploitation.
    Early in the search (high uncertainty), the cost penalty is reduced (alpha < 1),
    allowing the algorithm to explore more freely. Late in the search (low uncertainty),
    the cost penalty is increased (alpha > 1), strictly conserving budget.
    
    Args:
        mu: (2000,) array of GP mean predictions.
        sigma: (2000,) array of GP standard deviation predictions.
        f_best: float, the lowest observed value.
        cost: (2000,) array of evaluation costs.
        
    Returns:
        (2000,) array of acquisition scores.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        return np.zeros_like(mu)

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero or negative costs in cost calculations
    # Cost is defined as 1 + mean(x_scaled), so it should be >= 1.
    # But we guard against any numerical issues.
    cost_safe = np.maximum(cost, 1e-6)
    
    # Calculate Standard Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # We use a small epsilon to avoid division by zero in z
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    z = (f_best - mu) / sigma_safe
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # EI should be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Adaptive Cost Exponent Strategy
    # Measure the current level of uncertainty
    mean_sigma = np.mean(sigma)
    
    # Define thresholds for "early" and "late" search
    # In standardized BO, sigma typically starts around 1.0 and decreases.
    # We map mean_sigma to a coefficient t in [0, 1]
    # t = 1.0 means "very early" (high sigma), t = 0.0 means "very late" (low sigma)
    
    # Heuristic bounds: 
    # If mean_sigma > 0.5, consider it early.
    # If mean_sigma < 0.1, consider it late.
    # Linear interpolation between these points.
    
    sigma_high = 0.5
    sigma_low = 0.1
    
    if mean_sigma >= sigma_high:
        t = 1.0
    elif mean_sigma <= sigma_low:
        t = 0.0
    else:
        t = (mean_sigma - sigma_low) / (sigma_high - sigma_low)
        
    # t is in [0, 1]. t=1 is early, t=0 is late.
    
    # Alpha ranges from 0.4 (early, lenient cost) to 1.2 (late, strict cost)
    alpha_min = 0.4
    alpha_max = 1.2
    
    # alpha = alpha_min + (alpha_max - alpha_min) * (1 - t)
    # If t=1 (early): alpha = alpha_min = 0.4
    # If t=0 (late): alpha = alpha_max = 1.2
    alpha = alpha_min + (alpha_max - alpha_min) * (1.0 - t)
    
    # Calculate Cost-Aware Score
    # Score = EI / Cost^alpha
    # Using log-space for stability if needed, but direct power is fine for small arrays
    # Cost is >= 1, so Cost^alpha is stable.
    
    score = ei / (cost_safe ** alpha)
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
