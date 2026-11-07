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
    
    Uses a hybrid of Expected Improvement and Lower Confidence Bound,
    with adaptive exploration and cost normalization.
    
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
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        scores = ei / (cost + 1e-9)
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure inputs are finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid numerical issues
    sigma = np.clip(sigma, 1e-8, None)
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute Lower Confidence Bound (exploit + explore)
    # Use an adaptive beta that increases as we approach the budget limit
    # Total budget is 30, initial cost from 5 uniform points is roughly 5 * (1 + 0.5) = 7.5
    # So remaining budget is roughly 30 - current_cost
    # We estimate current cost from the mean of the cost array (rough proxy)
    mean_cost = np.mean(cost)
    # Estimate how much budget has been used: we don't know exactly, but we can use
    # the fact that each evaluation costs ~1.5 on average, and we've done some evaluations.
    # Actually, we don't have direct access to the number of evaluations done.
    # Let's use a different approach: beta based on the ratio of sigma to the range of mu
    mu_range = np.max(mu) - np.min(mu) + 1e-9
    sigma_mean = np.mean(sigma)
    
    # Adaptive beta: larger when uncertainty is high relative to the range
    # This encourages exploration when the model is uncertain
    beta = 1.0 + 2.0 * (sigma_mean / mu_range)
    beta = np.clip(beta, 0.5, 5.0)
    
    lcb = mu - beta * sigma
    
    # Combine EI and LCB with cost normalization
    # Normalize EI to [0, 1] roughly
    ei_max = np.max(ei) + 1e-9
    ei_norm = ei / ei_max
    
    # Normalize LCB to [0, 1] roughly (lower LCB is better, so negate)
    lcb_min = np.min(lcb)
    lcb_max = np.max(lcb) + 1e-9
    lcb_norm = (lcb_max - lcb) / (lcb_max - lcb_min + 1e-9)
    
    # Weighted combination: 60% EI, 40% LCB
    combined = 0.6 * ei_norm + 0.4 * lcb_norm
    
    # Cost-aware: divide by cost, with a floor to avoid division by near-zero
    cost_safe = np.clip(cost, 1e-6, None)
    scores = combined / cost_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
