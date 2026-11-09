import numpy as np
import time

# Global state for adaptive acquisition
_STEP = 0
_TOTAL_STEPS = 25  # 30 total evals - 5 initial

# Wall-clock deadline guard
_T0 = time.time()
_WALL_LIMIT = 330.0
_SAFETY = 45.0
_DEADLINE = _WALL_LIMIT - _SAFETY

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: combines Expected Improvement (EI) for exploitation
    with a time-varying exploration bonus based on posterior uncertainty.
    
    EI naturally balances exploration and exploitation based on f_best,
    while the exploration bonus ensures coverage of the multimodal landscape
    in early steps. The bonus decays as we accumulate more information.
    
    Args:
        mu: (2000,) array of posterior means.
        sigma: (2000,) array of posterior standard deviations.
        f_best: float, the lowest observed value.
        
    Returns:
        (2000,) array of scores. Higher score means evaluate next.
    """
    global _STEP
    
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback to simple greedy if time is running out
        return -mu
    
    # Update step counter
    _STEP += 1
    t = min(_STEP / _TOTAL_STEPS, 1.0)
    
    # Compute Expected Improvement for minimization
    # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x))/sigma(x)) + sigma(x) * phi((f_best - mu(x))/sigma(x))
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    from scipy.stats import norm
    
    # Avoid division by zero
    sigma_safe = np.where(sigma > 1e-10, sigma, 1e-10)
    
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a small exploration bonus that decays over time
    # This helps escape local optima in the multimodal Hartmann 6d landscape
    exploration_weight = 0.5 * (1.0 - t)
    exploration_bonus = exploration_weight * sigma
    
    # Combine EI and exploration bonus
    scores = ei + exploration_bonus
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
