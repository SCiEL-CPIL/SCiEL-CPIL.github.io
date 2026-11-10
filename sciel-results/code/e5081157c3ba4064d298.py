"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # seconds, with safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Normalized EI and Adaptive PI.
    
    This approach normalizes EI and PI to comparable scales, blends them with 
    weights that favor exploration (PI) in high-uncertainty regions, and applies 
    a relative cost penalty.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions
    sigma : array-like, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed function value
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate (1 + mean of scaled coords)
    
    Returns:
    --------
    score : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Deadline guard: if we are too close to the limit, return a fast, simple score
    if time.time() > _DEADLINE:
        # Fallback: Simple EI / cost
        sigma_safe = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-9)
        cost_safe = np.maximum(np.asarray(cost, dtype=np.float64), 1e-9)
        mu_safe = np.asarray(mu, dtype=np.float64)
        z = (f_best - mu_safe) / sigma_safe
        ei = (f_best - mu_safe) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        score = ei / cost_safe
        return np.where(np.isfinite(score), score, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    
    # Sanitize inputs
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Ensure positive sigma and cost
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # 1. Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical overflow in cdf/pdf
    z = np.clip(z, -100, 100)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # 2. Compute Probability of Improvement (PI)
    pi = norm.cdf(z)
    pi = np.where(np.isfinite(pi), pi, 0.0)
    
    # 3. Normalize EI and PI to [0, 1] range for comparable blending
    ei_max = np.max(ei)
    pi_max = np.max(pi)
    
    if ei_max < 1e-12:
        ei_norm = np.zeros_like(ei)
    else:
        ei_norm = ei / ei_max
        
    if pi_max < 1e-12:
        pi_norm = np.zeros_like(pi)
    else:
        pi_norm = pi / pi_max
        
    # 4. Adaptive Weighting
    # Higher uncertainty (sigma) -> higher weight on PI (exploration)
    # Lower uncertainty -> higher weight on EI (exploitation)
    # Normalize sigma to [0, 1] for weighting
    sigma_max = np.max(sigma)
    if sigma_max < 1e-9:
        uncertainty = np.zeros_like(sigma)
    else:
        uncertainty = sigma / sigma_max
        
    # Weight range: PI weight between 0.3 and 0.7
    pi_weight = 0.3 + 0.4 * uncertainty
    ei_weight = 1.0 - pi_weight
    
    # 5. Cost Weighting
    # Normalize cost to [0, 1] to make the penalty relative to the batch
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    
    if cost_range < 1e-12:
        cost_norm = np.zeros_like(cost)
    else:
        cost_norm = (cost - cost_min) / cost_range
        
    # Invert cost norm so lower cost gets higher weight
    # Adding a small epsilon to avoid division by zero for the cheapest point
    cost_weight = 1.0 / (cost_norm + 0.1)
    
    # 6. Combine
    score = (ei_weight * ei_norm + pi_weight * pi_norm) * cost_weight
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
