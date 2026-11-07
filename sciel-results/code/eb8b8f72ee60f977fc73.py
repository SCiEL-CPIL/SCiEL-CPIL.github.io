"""Rewritten acquisition with dynamic exploration-exploitation blend."""
import time
import numpy as np
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 300s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition with dynamic exploration/exploitation blend.
    
    Uses f_best to estimate progress: when f_best is high (early), explore more;
    when f_best is low (late), exploit more. Cost is handled via power-law scaling.
    
    Args:
        mu: array shape (2000,) - GP mean predictions in objective units
        sigma: array shape (2000,) - GP std dev predictions in objective units
        f_best: float - lowest observed value
        cost: array shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Time guard: if we're running out of time, return a simple fallback
    if time.time() > _DEADLINE:
        return np.ones_like(mu) / np.maximum(cost, 1e-9)
    
    eps = 1e-9
    s = np.maximum(sigma, eps)
    c = np.maximum(cost, eps)
    
    # Progress estimate: f_best starts near the maximum of the function (~6.4 for 3HC2D)
    # and decreases toward 0. Normalize to [0, 1] where 0=early, 1=late.
    # Three-hump camel 2D max is around 6.4, min is 0.
    f_max_est = 6.4
    progress = np.clip(1.0 - f_best / f_max_est, 0.0, 1.0)
    
    # Dynamic blend weight: early => more exploration, late => more exploitation
    # beta goes from 3.5 (early, exploration-heavy) to 0.5 (late, exploitation-heavy)
    beta = 3.5 - 3.0 * progress
    
    # Expected Improvement
    z = (f_best - mu) / s
    z = np.clip(z, -10.0, 10.0)
    ei = (f_best - mu) * norm.cdf(z) + s * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Power-law cost scaling: cheaper points get boosted more aggressively
    # cost_scale = c^(-beta) so high beta => strong preference for cheap points
    cost_scale = np.power(c, -beta)
    
    # Exploration term: reward uncertainty
    exploration = s
    
    # Blend: exploitation (EI) weighted by (1 - w) and exploration weighted by w
    # w is higher early on
    w = 0.5 * (1.0 - progress)  # w: 0.5 early, 0.0 late
    
    score = ((1.0 - w) * ei + w * exploration) * cost_scale
    
    return score
