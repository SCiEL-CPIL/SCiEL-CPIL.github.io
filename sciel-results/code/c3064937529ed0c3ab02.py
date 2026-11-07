"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_T0 = time.time()
_DEADLINE = 310.0


def _time_ok():
    return (time.time() - _T0) < _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Styblinski-Tang 2D.
    
    Structural change: Uses a pure cost-adjusted Upper Confidence Bound (UCB) for
    minimization, which directly optimizes an optimistic worst-case bound rather than
    expected improvement. This changes the exploration strategy from EI (which
    optimizes expected magnitude of improvement) to UCB (which optimizes an optimistic
    worst-case bound). UCB with a larger beta explores more aggressively in high-variance
    regions, potentially escaping local minima or finding better global minima in the
    Styblinski-Tang landscape, while the cost division maintains efficiency.
    
    The UCB for minimization is: UCB = mu - beta * sigma
    We want to maximize the acquisition score (higher = evaluate next), so we use
    -UCB = -mu + beta * sigma, normalized by cost.
    
    To keep scores bounded and well-scaled, we use:
    score = (f_best - mu + beta * sigma) / cost
    
    This ensures that:
    - Points with mu << f_best (predicted better than best) get high scores
    - Points with high sigma get boosted (exploration)
    - Expensive points are penalized by dividing by cost
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # UCB for minimization: we want to find points where the lower bound is low
    # UCB_min(x) = mu(x) - beta * sigma(x)
    # Acquisition score = -UCB_min(x) / cost(x) = (-mu + beta * sigma) / cost
    # 
    # But to keep scores in a reasonable range and ensure that points predicted
    # to be better than f_best are strongly favored, we use:
    # score = (f_best - mu + beta * sigma) / cost
    #
    # This is equivalent to: score = (f_best - (mu - beta*sigma)) / cost
    # which is the expected improvement over the UCB bound, divided by cost.
    
    # For low-budget multimodal optimization, use a larger beta to encourage
    # exploration of high-variance regions where the global minimum may lie
    beta = 2.0
    
    # Compute the cost-adjusted UCB score
    score = (f_best - mu + beta * sigma_safe) / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
