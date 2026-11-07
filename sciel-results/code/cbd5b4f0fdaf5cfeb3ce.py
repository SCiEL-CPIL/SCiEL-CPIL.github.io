import numpy as np
from scipy.stats import norm
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_DEADLINE_MARGIN = 5.0  # seconds to leave before the 330s kill
_START_TIME = None


def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a portfolio approach: computes Expected Improvement (EI) and
    Lower Confidence Bound (LCB) with adaptive exploration, then selects
    the maximum score across both strategies. Cost-awareness is handled
    via a dynamic exponent that increases as we approach the known minimum.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best (lowest) observed objective value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x scaled to [0,1])).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = prefer to evaluate next).
    """
    # Deadline guard: if we're getting close to the limit, return a safe fallback
    if _elapsed() > 330 - _DEADLINE_MARGIN:
        # Fallback: simple EI/cost, but ensure finite
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        scores = ei / np.maximum(cost, 1e-9)
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure inputs are finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Clip sigma to avoid numerical issues
    sigma = np.clip(sigma, 1e-6, 1e2)
    
    # Dynamic cost exponent: more cost-sensitive as we approach the optimum
    f_min = -0.8
    f_max = 1.0
    f_best_clipped = np.clip(f_best, f_min, f_max)
    t = (f_best_clipped - f_min) / (f_max - f_min)  # t=1 early, t=0 late
    alpha = 1.0 + (1.0 - t) * 1.0  # alpha = 2 - t, ranges [1, 2]
    
    cost_safe = np.maximum(cost, 1e-9)
    cost_factor = np.power(cost_safe, alpha)
    
    # 1. Expected Improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    score_ei = ei / cost_factor
    
    # 2. Lower Confidence Bound (LCB) with adaptive beta
    # Adaptive beta: larger when uncertainty is high relative to the range
    mu_range = np.max(mu) - np.min(mu) + 1e-9
    sigma_mean = np.mean(sigma)
    sigma_max = np.max(sigma)
    
    # Combine two beta estimation strategies
    beta_from_range = 1.0 + 2.0 * (sigma_mean / mu_range)
    beta_from_max = 1.5 + 2.0 * (sigma_max / (mu_range + 1e-9))
    beta = np.clip(0.5 * beta_from_range + 0.5 * beta_from_max, 0.5, 5.0)
    
    lcb = mu - beta * sigma
    # Negate LCB so higher is better
    score_lcb = (-lcb) / cost_factor
    
    # Portfolio: take the maximum across strategies
    scores = np.maximum(score_ei, score_lcb)
    
    # Ensure all outputs are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
