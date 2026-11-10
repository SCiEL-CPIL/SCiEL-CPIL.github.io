"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 315.0  # Safety margin before 330s hard kill

def _value_function(mu, sigma, f_best, cost, budget_remaining):
    """
    Approximate dynamic programming value function for cost-aware acquisition.
    
    The idea: we estimate the probability that evaluating point x will lead to
    finding a better solution within the remaining budget. We approximate this
    by considering:
    1. The immediate improvement potential (f_best - mu)
    2. The uncertainty (sigma) which represents exploration value
    3. The cost of evaluation relative to remaining budget
    
    We model this as: V(x) = P(improvement) * E[improvement] / cost(x)
    where P(improvement) is approximated by the CDF of the improvement distribution,
    and we add an exploration bonus that decays with cost.
    
    This is a structural departure from static EI/UCB: it considers the trajectory
    of future evaluations by weighting current improvement by the likelihood of
    being able to continue optimizing effectively.
    """
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score for improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of improvement (P(f(x) < f_best))
    # This is Phi(z) where z = (f_best - mu) / sigma
    p_improve = norm.cdf(z)
    
    # Expected improvement given that improvement occurs
    # E[f_best - f(x) | f(x) < f_best]
    # This is (f_best - mu) * Phi(z) + sigma * phi(z) divided by Phi(z) when Phi(z) > 0
    # But we want unconditional expected improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-to-go: how many more evaluations can we afford?
    # If budget_remaining is the total cost left, and cost(x) is the cost of x,
    # then after evaluating x, we have budget_remaining - cost(x) left.
    # The value of evaluating x is higher if it leaves more budget for future improvements.
    
    # Approximate the "value of information" by considering:
    # 1. Immediate expected improvement
    # 2. The fraction of budget consumed (higher cost = less future flexibility)
    
    # Budget efficiency: how much improvement per unit cost
    # We also add an exploration term that accounts for the value of reducing uncertainty
    # The exploration value is higher when:
    # - sigma is large (high uncertainty)
    # - cost is low (leaves more budget)
    # - we're early in the process (more budget remaining)
    
    # Normalize cost to [0, 1] range for scaling
    cost_min = np.min(cost_safe)
    cost_max = np.max(cost_safe)
    cost_range = cost_max - cost_min
    if cost_range < 1e-9:
        cost_norm = np.ones_like(cost_safe)
    else:
        cost_norm = (cost_safe - cost_min) / cost_range
    
    # Budget remaining ratio: assume budget is around 30 total
    # Estimate how much budget is left based on current f_best
    # Early: f_best is large, lots of budget left
    # Late: f_best is small, little budget left
    # Use a heuristic: budget_fraction = clip(1 - f_best/10, 0.1, 1.0)
    budget_fraction = np.clip(1.0 - f_best / 10.0, 0.1, 1.0)
    
    # Exploration bonus: value of reducing uncertainty
    # Scaled by budget fraction (more exploration value when we have budget)
    exploration_bonus = sigma_safe * budget_fraction * (1.0 - cost_norm)
    
    # Combine: immediate improvement per cost + exploration bonus
    # The immediate improvement is weighted by the probability of improvement
    # This creates a "trajectory-aware" score that considers both now and future
    
    # Score = (immediate improvement * P(improve)) / cost + exploration bonus
    # But we want to ensure that points with high cost but very high potential are still considered
    # Use a soft penalty for cost rather than hard division
    
    # Alternative: use a power law for cost penalty
    # score = (ei + exploration_bonus) / cost^alpha
    # where alpha < 1 makes the cost penalty less severe
    
    alpha = 0.7  # Softer cost penalty than pure 1/cost
    score = (ei + exploration_bonus) / np.power(cost_safe, alpha)
    
    return score

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a value-function-inspired approach.
    
    This models the optimization as a stochastic process where the value of
    evaluating a point depends not just on immediate improvement but on the
    trajectory of future evaluations possible within the remaining budget.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far (lower is better).
    cost : np.ndarray, shape (2000,)
        Evaluation cost for each candidate.
    
    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = better to evaluate next).
    """
    # Time guard
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        fallback = ei / cost_safe
        return np.where(np.isfinite(fallback), fallback, 0.0)
    
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Compute the value-function-based acquisition score
    score = _value_function(mu, sigma, f_best, cost, budget_remaining=30.0)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
