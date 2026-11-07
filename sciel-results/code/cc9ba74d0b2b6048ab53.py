import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_start_time = time.time()
_DEADLINE = _start_time + 320.0  # 320s wall-clock safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions
    sigma : array-like, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed objective value (lowest)
    cost : array-like, shape (2000,)
        Evaluation cost for each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-8)
    
    # Deadline check - if we're close to deadline, return a simple score
    if time.time() > _DEADLINE - 5.0:
        # Return a simple, fast score
        z = (f_best - mu) / sigma
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        scores = ei / cost
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute Probability of Improvement (PI)
    pi = norm.cdf(z)
    
    # Compute Lower Confidence Bound (LCB) desirability for minimization
    # LCB value = mu - beta * sigma (lower is better for minimization)
    # Desirability = -LCB value = -mu + beta * sigma
    # Use adaptive beta that shifts from exploration to exploitation based on cost budget
    
    # Estimate remaining budget ratio: as cost increases, shift to exploitation
    # The total cost budget is 30, so remaining = 1 - cost/30 (clipped)
    # But cost here is per-candidate, not total. Use f_best and sigma to gauge progress.
    
    # Adaptive beta: start high (exploration) and decrease (exploitation)
    # Use the spread of sigma to determine exploration level
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma)
    
    # If sigma is high relative to f_best, we're in early exploration
    # If sigma is low, we should exploit
    if f_best > 1e-10:
        exploration_ratio = sigma_mean / (f_best + 1e-10)
    else:
        exploration_ratio = 1.0
    
    # Clip exploration ratio to [0, 1]
    exploration_ratio = np.clip(exploration_ratio, 0.0, 1.0)
    
    # Beta ranges from 2.0 (high exploration) to 0.3 (high exploitation)
    beta = 0.3 + 1.7 * exploration_ratio
    
    lcb_desirability = -mu + beta * sigma
    
    # Normalize EI and LCB to comparable scales
    ei_max = np.max(ei)
    if ei_max > 1e-10:
        ei_norm = ei / ei_max
    else:
        ei_norm = ei
    
    lcb_max = np.max(lcb_desirability)
    lcb_min = np.min(lcb_desirability)
    if lcb_max - lcb_min > 1e-10:
        lcb_norm = (lcb_desirability - lcb_min) / (lcb_max - lcb_min)
    else:
        lcb_norm = lcb_desirability
    
    # Combine: weighted sum of normalized EI and LCB
    # EI is the primary signal, LCB provides exploration
    # Shift weights from LCB to EI as exploration_ratio decreases
    weights_ei = 0.5 + 0.3 * (1.0 - exploration_ratio)
    weights_lcb = 1.0 - weights_ei
    
    combined = weights_ei * ei_norm + weights_lcb * lcb_norm
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    scores = combined / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure non-negative scores (acquisition should be non-negative)
    scores = np.maximum(scores, 0.0)
    
    return scores
