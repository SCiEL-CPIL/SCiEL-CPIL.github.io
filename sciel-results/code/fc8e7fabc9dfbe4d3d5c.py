"""Cost-aware acquisition function for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: the evaluator kills the program after 330s wall-clock.
_START_TIME = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 5.0


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME) - _SAFETY_MARGIN


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function blending Expected Improvement and
    Probability of Improvement, normalized and divided by evaluation cost.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best (lowest) observed objective value so far.
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate: 1 + mean(x) where x is in [0,1]^3.
    
    Returns
    -------
    scores : np.ndarray, shape (2000,)
        Acquisition scores; higher values are preferred.
    """
    # Check deadline: if we're running out of time, return a simple fallback
    if _time_left() <= 0:
        return -mu / (cost + 1e-9)
    
    # Ensure sigma and cost are positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement (EI) for minimization
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # Clip z to avoid numerical issues with extreme values
    z = np.clip(z, -50, 50)
    
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement (PI) for minimization
    pi = norm.cdf(z)
    
    # Normalize EI and PI to [0, 1] using max-based normalization
    ei_max = np.max(ei)
    pi_max = np.max(pi)
    
    if ei_max > 0:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    if pi_max > 0:
        pi_norm = pi / pi_max
    else:
        pi_norm = np.zeros_like(pi)
    
    # Blend EI and PI: 70% exploitation (EI), 30% exploration (PI)
    alpha = 0.7
    blended = alpha * ei_norm + (1 - alpha) * pi_norm
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    scores = blended / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
