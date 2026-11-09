"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Rastrigin 2D.
    
    Uses Probability of Improvement (PI) as the primary signal, which proved
    more robust than EI in this setting. PI is less sensitive to the magnitude
    of potential improvement, which can be misleading when the GP's scale is
    not yet well-calibrated. The 1/cost normalization effectively balances
    the trade-off between exploration probability and evaluation expense.
    
    A small UCB exploration term is blended in during early stages to help
    escape local minima in the multimodal Rastrigin landscape. The blend
    shifts from exploration-heavy to exploitation-heavy as f_best improves.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions at 2000 candidate points.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions at 2000 candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next).
    """
    # Deadline guard
    if time.time() - _start_time > 320:
        return np.zeros_like(mu)
    
    # Guard against division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Probability of Improvement (PI)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # UCB score for minimization: -mu + beta * sigma
    # Beta adapts: higher when f_best is large (more exploration needed)
    f_best_norm = np.clip(f_best / 20.0, 0.0, 1.0)
    beta = 0.5 + 1.5 * f_best_norm
    
    ucb = -mu + beta * sigma_safe
    
    # Normalize PI and UCB to comparable scales
    # PI is in [0, 1], UCB can be large negative or positive
    # Scale UCB by typical PI magnitude to make them comparable
    pi_mean = np.mean(pi)
    if pi_mean > 1e-10:
        ucb_scaled = ucb * pi_mean / (np.std(ucb) + 1e-10)
    else:
        ucb_scaled = np.zeros_like(ucb)
    
    # Adaptive blend: PI-dominant, with UCB providing exploration help
    # When f_best is high, weight UCB more; when close to 0, weight PI more
    pi_weight = 0.7 + 0.3 * (1.0 - f_best_norm)  # 1.0 when f_best~0, 0.7 when f_best~20
    ucb_weight = 1.0 - pi_weight                  # 0.0 when f_best~0, 0.3 when f_best~20
    
    combined = pi_weight * pi + ucb_weight * ucb_scaled
    
    # Cost-aware: divide by cost
    scores = combined / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
