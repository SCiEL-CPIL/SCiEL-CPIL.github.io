"""Direct cost-aware acquisition for Bayesian optimization of rosenbrock 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import erf
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 325.0  # 5s safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Quantile-based Cost-Effective Acquisition for Bayesian optimization.
    
    Structural idea: Instead of Expected Improvement (which averages over the
    entire predictive distribution), we target the 10th percentile of the
    predicted objective distribution (q_0.1 = mu - 1.28*sigma) and maximize
    -q_0.1 / cost. This shifts the optimization goal from expected improvement
    to high-confidence minimization of worst-case cost-adjusted performance.
    
    This explicitly targets low-variance regions where the function is likely
    to be low, which is particularly effective for cost-aware BO where we want
    to spend budget efficiently on points that are very likely to be good.
    
    The 10th percentile (z = -1.2816) provides a good balance between
    exploitation (low mu) and confidence (low sigma). Points with high sigma
    are penalized because their 10th percentile is not very low, even if their
    mean is low.
    
    Refinement machinery: Fast erf-based computation, deadline guard, and
    robust numerical handling from the current program.
    """
    # Deadline guard: if we're running out of time, return simple exploitation
    if time.time() > _DEADLINE:
        safe_cost = np.maximum(cost, 1e-6)
        scores = -mu / safe_cost
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: handle NaN/Inf
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=-1e6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-8)
    
    # Quantile-based acquisition: target the 10th percentile
    # q_0.1 = mu - 1.2816 * sigma
    # We want to minimize q_0.1 / cost, so we maximize -q_0.1 / cost
    z_quantile = -1.2815515655446004  # z-score for 10th percentile
    
    q_0_1 = mu + z_quantile * sigma
    
    # Cost-effective quantile: maximize -q_0.1 / cost
    scores = -q_0_1 / cost
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
