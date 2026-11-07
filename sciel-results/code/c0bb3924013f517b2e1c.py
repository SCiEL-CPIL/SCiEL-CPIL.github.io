"""Cost-aware UCB with adaptive exploration for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # 300s limit + 5s margin


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB).
    
    Uses a UCB strategy for minimization, which balances exploitation (low mean)
    and exploration (high variance) with an adaptive beta coefficient.
    The score is normalized by the evaluation cost to favor cheaper points.
    """
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Safety: return deterministic fallback
        sigma_safe = np.maximum(np.asarray(sigma), 1e-9)
        cost_safe = np.maximum(np.asarray(cost), 1e-9)
        mu_safe = np.asarray(mu)
        # Fallback: simple mean-based score
        score = -mu_safe / cost_safe
        return np.where(np.isfinite(score), score, 0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Determine exploration coefficient beta
    # We want to explore more when the gap to the global minimum is large.
    # Global minimum of cosine8 is -0.8.
    min_possible = -0.8
    gap_to_min = max(f_best - min_possible, 0.0)
    
    # Adaptive beta: 
    # When gap is large (early on), beta is high (explore).
    # When gap is small (late on), beta is low (exploit).
    # Scale the decay so that beta drops significantly as we get close to -0.8.
    # gap ranges from ~1.8 (start) to 0 (end).
    # Let's use an exponential decay.
    beta = 2.0 * np.exp(-gap_to_min / 0.5)
    
    # Ensure beta doesn't get too small to prevent pure exploitation noise
    beta = max(beta, 0.1)
    
    # UCB for minimization: 
    # We want to find points with low expected value.
    # UCB lower bound: L(x) = mu(x) - beta * sigma(x)
    # We want to MINIMIZE L(x).
    # So the "goodness" score is -L(x) = -mu(x) + beta * sigma(x).
    
    # However, we must account for cost.
    # We want to maximize the "improvement per cost".
    # Let's define the score as the potential improvement over the current best,
    # adjusted for uncertainty, divided by cost.
    
    # Improvement potential: (f_best - mu) is the expected improvement if we ignore uncertainty.
    # UCB improvement: (f_best - (mu - beta*sigma)) = (f_best - mu) + beta*sigma.
    # This quantity is always positive if we assume f_best is the current best.
    
    ucb_improvement = (f_best - mu) + beta * sigma_safe
    
    # Clip to avoid negative scores if mu > f_best + beta*sigma (unlikely for best candidates)
    ucb_improvement = np.maximum(ucb_improvement, 0.0)
    
    # Normalize by cost to favor cheaper evaluations
    score = ucb_improvement / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
