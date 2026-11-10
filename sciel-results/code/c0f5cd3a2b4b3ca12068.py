"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 320.0  # Safety margin before 330s hard kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    and Upper Confidence Bound (UCB) signals, normalized by evaluation cost.
    
    This approach balances exploitation (finding points likely to improve
    on the current best) with exploration (visiting uncertain regions)
    while respecting the non-uniform cost structure of the problem.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far (lower is better).
    cost : np.ndarray, shape (2000,)
        Evaluation cost for each candidate.
    
    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = better to evaluate next).
    """
    # Time guard: if we're running low on time, return a simple fallback
    if time.time() - _start_time > _TIME_LIMIT:
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        # Fallback: Pure EI / Cost
        fallback = ei / cost_safe
        return np.where(np.isfinite(fallback), fallback, 0.0)
    
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # 1. Expected Improvement (EI)
    # EI is the expected improvement over f_best.
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Adaptive UCB: scale exploration based on uncertainty relative to best
    # When sigma is large relative to the current best, increase exploration.
    # When we're close to optimum (small sigma, mu near f_best), reduce exploration.
    sigma_mean = np.mean(sigma)
    sigma_max = np.max(sigma)
    
    # Adaptive beta: higher when overall uncertainty is high, lower when converging
    # Normalize sigma by its max to get a relative uncertainty measure
    rel_uncertainty = np.clip(sigma / (sigma_max + 1e-9), 0.0, 1.0)
    
    # Base exploration weight that scales with relative uncertainty
    # Early: high uncertainty -> more exploration. Late: low uncertainty -> more exploitation.
    beta_base = 0.3 + 0.4 * rel_uncertainty  # ranges from 0.3 to 0.7
    
    ucb = -mu + beta_base * sigma
    
    # 3. Combined acquisition: EI-dominant with adaptive UCB
    # Weight EI more heavily for direct improvement, UCB for exploration
    # Use a smooth combination that avoids cancellation issues
    score_raw = ei + 0.6 * ucb
    
    # 4. Cost-aware normalization with slight convexity adjustment
    # Divide by cost but add a small constant to avoid extreme values at low cost
    cost_floor = 1.0  # Minimum effective cost to prevent division by very small costs
    score = score_raw / np.maximum(cost, cost_floor)
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Clip extreme scores to maintain numerical stability
    score_max = np.max(np.abs(score))
    if score_max > 0:
        # Scale to reasonable range to prevent overflow in downstream comparisons
        score = np.clip(score, -score_max, score_max)
    
    return score
# EVOLVE-BLOCK-END
