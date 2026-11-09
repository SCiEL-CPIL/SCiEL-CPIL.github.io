"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_ACQ_START = time.time()
_ACQ_DEADLINE = _ACQ_START + 280.0  # safety margin before 330s kill


def _pi(mu, sigma, f_best):
    """Probability of Improvement: P(f(x) < f_best | GP)."""
    z = (f_best - mu) / (sigma + 1e-9)
    return norm.cdf(z)


def _ucb_score(mu, sigma, f_best, beta):
    """Lower confidence bound (for minimization): mu - beta * sigma."""
    return mu - beta * sigma


def acquisition(mu, sigma, f_best, cost):
    # Time guard: if we're running long, fall back to a simple score
    if time.time() > _ACQ_DEADLINE:
        return -mu / (cost + 1e-9)
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # PI: probability that this point improves on f_best
    pi_scores = _pi(mu, sigma_safe, f_best)
    
    # UCB with adaptive beta: more exploration when uncertainty is high
    # Scale beta based on relative uncertainty
    sigma_mean = np.mean(sigma_safe)
    sigma_std = np.std(sigma_safe) + 1e-9
    beta = 2.0 + 1.5 * (sigma_std / (sigma_mean + 1e-9))
    
    ucb_scores = _ucb_score(mu, sigma_safe, f_best, beta)
    
    # Normalize both to [0, 1] for blending
    pi_min, pi_max = np.min(pi_scores), np.max(pi_scores)
    pi_range = pi_max - pi_min + 1e-9
    pi_norm = (pi_scores - pi_min) / pi_range
    
    ucb_min, ucb_max = np.min(ucb_scores), np.max(ucb_scores)
    ucb_range = ucb_max - ucb_min + 1e-9
    ucb_norm = (ucb_scores - ucb_min) / ucb_range
    
    # Blend: 60% PI, 40% UCB
    blended = 0.6 * pi_norm + 0.4 * ucb_norm
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    # Add small offset to avoid division by zero
    cost_safe = np.maximum(cost, 1e-6)
    score = blended / cost_safe
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
