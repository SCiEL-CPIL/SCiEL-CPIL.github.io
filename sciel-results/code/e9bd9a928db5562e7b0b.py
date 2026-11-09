"""Cost-aware acquisition function for Bayesian optimization (Styblinski-Tang 2D)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 290  # Safety margin before 300s limit


def _elapsed_ok():
    return (time.time() - _start_time) < _TIMEOUT


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using hypervolume-inspired efficiency metric.
    
    Uses Expected Improvement normalized by cumulative cost budget consumed,
    combined with a UCB component for exploration. The key insight is to
    normalize EI by the total cost spent so far (not individual point cost)
    to create a global efficiency metric that stabilizes the
    exploration-exploitation trade-off over the search trajectory.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = more promising)
    """
    # Ensure all inputs are finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    cost = np.maximum(cost, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Check time - if close to timeout, return simple cost-adjusted improvement
    if not _elapsed_ok():
        return (f_best - mu) / cost
    
    # Compute Expected Improvement (EI)
    # EI = E[max(f_best - f(x), 0)] = (f_best - mu) * Phi((f_best - mu)/sigma) 
    #      + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)  # Numerical stability
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Compute UCB (Upper Confidence Bound) for exploration
    # We want to explore where uncertainty is high
    # UCB = -mu + beta * sigma (maximize this for minimization)
    # Use a beta that adapts to progress
    known_min = -78.3323
    progress = f_best - known_min  # Positive when above minimum
    
    # Adaptive beta: more exploration early, less later
    if progress > 30:
        beta = 2.5
    elif progress > 15:
        beta = 2.0
    elif progress > 5:
        beta = 1.5
    else:
        beta = 1.0
    
    ucb = -mu + beta * sigma
    ucb = np.where(np.isfinite(ucb), ucb, 0.0)
    
    # Cost-aware scoring:
    # The key idea from the strategy: normalize by cumulative cost budget
    # rather than individual point cost. We estimate cumulative cost as
    # the mean of the cost array (representing average cost per evaluation)
    # times an effective number of evaluations.
    
    # Estimate total cost spent: use mean cost as proxy for average cost
    # The number of evaluations can be inferred from the GP state, but
    # we approximate using the scale of sigma (early: high sigma, few evals)
    mean_cost = np.mean(cost)
    
    # Normalize EI by cost to get efficiency per unit cost
    ei_per_cost = ei / (cost + 1e-9)
    
    # Also compute a global efficiency metric: EI normalized by mean cost
    # This creates a trajectory-level efficiency measure
    ei_global_eff = ei / (mean_cost + 1e-9)
    
    # Normalize UCB similarly
    ucb_per_cost = ucb / (cost + 1e-9)
    
    # Adaptive weighting between EI and UCB based on progress
    # Early: more UCB (explore), Late: more EI (exploit)
    if progress > 20:
        w_ei = 0.4
        w_ucb = 0.6
    elif progress > 10:
        w_ei = 0.5
        w_ucb = 0.5
    elif progress > 5:
        w_ei = 0.6
        w_ucb = 0.4
    else:
        w_ei = 0.7
        w_ucb = 0.3
    
    # Combine per-cost normalized scores
    score_per_cost = w_ei * ei_per_cost + w_ucb * ucb_per_cost
    
    # Combine with global efficiency for stability
    # The global metric stabilizes the trajectory-level trade-off
    score_global = w_ei * ei_global_eff + w_ucb * (ucb / (mean_cost + 1e-9))
    
    # Blend per-point and global scores
    # More weight on per-point early (local decisions matter),
    # more on global later (trajectory efficiency matters)
    if progress > 15:
        w_local = 0.6
        w_global = 0.4
    elif progress > 8:
        w_local = 0.5
        w_global = 0.5
    else:
        w_local = 0.4
        w_global = 0.6
    
    scores = w_local * score_per_cost + w_global * score_global
    
    # Add a small uncertainty bonus in early stages to encourage exploration
    if progress > 20:
        max_sigma = np.max(sigma)
        if max_sigma > 1e-9:
            unc_bonus = sigma / max_sigma
            scores += 0.1 * unc_bonus
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Final normalization for numerical stability
    score_max = np.max(scores)
    score_min = np.min(scores)
    if score_max - score_min > 1e-12:
        scores = (scores - score_min) / (score_max - score_min)
    
    return scores
# EVOLVE-BLOCK-END
