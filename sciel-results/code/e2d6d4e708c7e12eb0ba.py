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
    Cost-aware acquisition function blending Lower Confidence Bound (LCB) 
    and Probability of Improvement (PI).
    
    This structural change moves from an EI-dominant strategy to an LCB-dominant 
    strategy, which provides a more linear exploration/exploitation trade-off 
    and is often more robust in cost-aware settings where the cost penalty 
    needs to be balanced against the confidence interval width.
    
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
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / cost_safe
    
    # Ensure sigma and cost are positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # 1. Compute Lower Confidence Bound (LCB) for minimization
    # LCB = -mu + beta * sigma
    # We use beta = 2.0 as a robust default for 3D problems
    beta = 2.0
    lcb = -mu + beta * sigma_safe
    
    # 2. Compute Probability of Improvement (PI)
    improvement = f_best - mu
    z = improvement / sigma_safe
    z = np.clip(z, -50, 50)
    pi = norm.cdf(z)
    
    # 3. Normalize LCB to [0, 1] range
    # Min-max normalization with safety epsilon
    lcb_min = np.min(lcb)
    lcb_max = np.max(lcb)
    lcb_range = lcb_max - lcb_min
    
    if lcb_range > 1e-9:
        lcb_norm = (lcb - lcb_min) / lcb_range
    else:
        lcb_norm = np.zeros_like(lcb)
    
    # 4. Blend LCB and PI
    # We weight LCB and PI equally to balance exploration (LCB) and 
    # likelihood of improvement (PI). PI helps avoid regions that are 
    # uncertain but likely to be poor (low PI).
    alpha = 0.5
    blended = alpha * lcb_norm + (1 - alpha) * pi
    
    # 5. Cost-aware: divide by cost to prefer cheaper evaluations
    # Using a slight power on cost (1.2) to penalize expensive points more strongly
    # can help navigate the cost landscape better in Hartmann 3d.
    scores = blended / (cost_safe ** 1.2)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
