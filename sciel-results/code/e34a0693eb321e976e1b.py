"""Rewritten program: direct cost-aware acquisition for Rastrigin 2D Bayesian optimization."""
import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE_SECONDS = 320.0  # Safety margin before 330s wall-clock limit

def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        return np.zeros_like(mu, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=-1e6)
    cost = np.nan_to_num(cost, nan=1.5, posinf=2.0, neginf=1.0)
    
    # Cost is in [1, 2] (1 + mean of x scaled to [0,1])
    cost = np.clip(cost, 1.0, 2.0)
    sigma = np.maximum(sigma, 1e-12)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    z = np.clip(z, -50, 50)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Upper Confidence Bound for minimization
    # UCB = -mu + kappa * sigma
    # We maximize this score.
    # Cost-aware: divide by cost to prefer cheaper points.
    # Typical values for UCB in BO are 1-5.
    kappa = 2.0
    ucb = -mu + kappa * sigma
    
    # Blend EI and UCB: EI captures exploitation, UCB captures exploration
    # Weight them to balance exploitation and exploration
    # EI is bounded by (f_best - min(mu)) roughly, UCB can be unbounded
    # Normalize both to have similar scale
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ucb_max = np.max(ucb) if np.max(ucb) > 0 else 1.0
    
    # Blend: 0.5 * EI + 0.5 * UCB, normalized to similar scale
    blended = 0.5 * (ei / ei_max) + 0.5 * (ucb / ucb_max)
    
    # Scale back to original magnitude
    blended = blended * np.maximum(ei_max, ucb_max)
    
    # Cost-aware normalization
    # Use blended / cost^alpha where alpha > 1 amplifies cost preference
    # alpha=2 gives range [blended/4, blended/1] instead of [blended/2, blended/1]
    # This makes the difference between low and high cost regions more pronounced
    alpha = 2.0
    score = blended / (cost ** alpha)
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
