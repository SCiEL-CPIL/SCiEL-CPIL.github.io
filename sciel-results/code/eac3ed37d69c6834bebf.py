import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 300.0  # safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of hartmann 3d.
    
    Strategy:
    1. Compute Expected Improvement (EI) and Probability of Improvement (PI).
    2. Normalize both to [0, 1] range based on current batch statistics.
    3. Blend them (50/50) to balance exploitation (EI) and exploration (PI).
    4. Add a UCB-like exploration bonus that scales with uncertainty.
    5. Scale by 1/cost to account for evaluation cost.
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        # Return a simple fallback: negative mean divided by cost
        return -mu / np.maximum(cost, 1e-9)
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-9)
    
    # Ensure cost is positive and finite
    cost_safe = np.maximum(cost, 1e-9)
    cost_safe = np.where(np.isfinite(cost_safe), cost_safe, 1.0)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # z-score for improvement
    z = (f_best - mu) / sigma_safe
    # Clip z for numerical stability
    z = np.clip(z, -10, 10)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Normalize EI and PI to [0, 1] for blending
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range < 1e-12:
        ei_norm = np.zeros_like(ei)
    else:
        ei_norm = (ei - ei_min) / ei_range
    
    pi_min = np.min(pi)
    pi_max = np.max(pi)
    pi_range = pi_max - pi_min
    if pi_range < 1e-12:
        pi_norm = np.zeros_like(pi)
    else:
        pi_norm = (pi - pi_min) / pi_range
    
    # Blend EI and PI
    # 50% weight to each. PI helps explore uncertain regions where EI might be low
    # due to high sigma but high potential.
    blended = 0.5 * ei_norm + 0.5 * pi_norm
    
    # UCB-like exploration bonus: scale by normalized sigma
    # This encourages exploration in high-uncertainty regions
    sigma_min = np.min(sigma_safe)
    sigma_max = np.max(sigma_safe)
    sigma_range = sigma_max - sigma_min
    if sigma_range < 1e-12:
        sigma_norm = np.zeros_like(sigma_safe)
    else:
        sigma_norm = (sigma_safe - sigma_min) / sigma_range
    
    # Dynamically tune kappa based on the spread of sigma
    # Higher spread in sigma suggests more exploration is needed
    kappa = 0.3 + 0.4 * np.mean(sigma_norm)
    
    # Add UCB component to the blended score
    ucb_bonus = kappa * sigma_norm
    combined = blended + ucb_bonus
    
    # Cost-aware score: combined score per unit cost
    score = combined / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
