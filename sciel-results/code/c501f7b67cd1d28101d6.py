"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for deadline guard
_T0 = [None]
_CALLS = [0]


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining EI with a time-aware exploration bonus.
    
    Args:
        mu: GP means, shape (2000,)
        sigma: GP std devs, shape (2000,)
        f_best: best observed value (scalar)
        cost: evaluation costs, shape (2000,)
        
    Returns:
        scores, shape (2000,), higher = evaluate next
    """
    # Deadline guard
    if _T0[0] is None:
        _T0[0] = time.time()
    _CALLS[0] += 1
    elapsed = time.time() - _T0[0]
    
    # Safety margin: 330s limit, stop heavy compute at 315s
    if elapsed > 315.0:
        s = np.maximum(sigma, 1e-9)
        c = np.maximum(cost, 1e-9)
        score = s / c
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    s = np.maximum(sigma, 1e-9)
    c = np.maximum(cost, 1e-9)
    
    # Expected Improvement (minimization): EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / s
    # Clip z to avoid numerical issues
    z = np.clip(z, -50.0, 50.0)
    
    Phi = norm.cdf(z)
    phi = norm.pdf(z)
    
    ei = (f_best - mu) * Phi + s * phi
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI
    ei_cost = ei / c
    
    # Adaptive exploration: use UCB-like term with uncertainty
    # Weight decreases with elapsed time (more exploitation later)
    # Use a moderate exploration weight that decays over time
    t_frac = min(elapsed / 280.0, 1.0)
    expl_weight = 0.3 * (1.0 - t_frac)
    
    # UCB term: sigma / cost
    ucb = s / c
    
    # Combine: cost-EI + exploration
    score = ei_cost + expl_weight * ucb
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
