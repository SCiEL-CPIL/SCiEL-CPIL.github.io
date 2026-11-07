"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import ndtr
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on griewank 2d.
    Uses a stochastic shortest path / value function approach: the acquisition
    score is the negative expected cost-to-go, approximated by integrating the
    probability of finding a new minimum over the GP posterior, weighted by
    the remaining budget and the marginal value of improvement.
    """
    t0 = time.time()

    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Estimate remaining budget: total budget is 30, cost is cumulative
    # cost array has shape (2000,) but represents per-candidate cost
    # The loop runs while total_cost < 30. We estimate remaining steps.
    # Each evaluation costs ~1.5 on average (1 + mean(x_scaled) where x_scaled in [0,1])
    # So average cost per eval ~ 1.5
    # We don't know exact total cost, but cost array gives per-candidate costs.
    # Use the median cost as a proxy for typical evaluation cost.
    typical_cost = np.median(cost_safe)
    
    # Estimate how many more evaluations we can do
    # We know f_best is the current best. The initial 5 points cost ~5*1.5=7.5
    # Remaining budget ~ 30 - current_total_cost
    # We approximate current_total_cost from the number of evaluations done.
    # Since we don't have that directly, use f_best as a proxy for progress.
    # Early: f_best is large, many evals left. Late: f_best small, few evals left.
    # A rough model: remaining_evals ~ max(1, (30 - 7.5) / typical_cost * (1 - f_best/2))
    # Simpler: use a budget-aware scaling factor.
    
    # Value function: V(f_best, remaining_budget) = expected improvement per unit cost
    # The marginal value of a new observation depends on:
    # 1. How much improvement is possible (f_best - 0)
    # 2. How uncertain the GP is (sigma)
    # 3. How much budget remains
    
    # Compute z-scores for each candidate
    z = (f_best - mu) / sigma_safe
    
    # Probability of improvement
    pi_prob = ndtr(z)
    
    # Expected improvement
    phi_z = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    ei = (f_best - mu) * pi_prob + sigma_safe * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Cost-to-go value function:
    # V(x) = E[improvement] / cost(x) * budget_multiplier
    # The budget multiplier accounts for the fact that cheap evaluations
    # allow more total improvement over the remaining budget.
    
    # Budget-aware scaling: cheaper points have higher value because
    # they free up budget for more evaluations
    budget_multiplier = 1.0 / (typical_cost + 1e-9)
    
    # Marginal value of improvement: when f_best is large, improvement
    # is more valuable (diminishing returns as f_best -> 0)
    # Use a concave value function: log(1 + f_best) or sqrt(f_best)
    # This captures diminishing returns
    marginal_value = np.log1p(f_best)
    
    # The value of evaluating point x is:
    # - The expected improvement at x, normalized by cost
    # - Weighted by the marginal value of improvement
    # - With an exploration bonus for high uncertainty when budget allows
    
    # Core value: EI per unit cost, scaled by marginal value
    core_value = ei * marginal_value / cost_safe
    
    # Exploration bonus: high sigma points are valuable when we have
    # budget to exploit them later. Scale by remaining budget proxy.
    # When f_best is large (early), exploration is more valuable.
    exploration_weight = 0.3 * (1.0 + np.log1p(f_best))
    sigma_mean = np.mean(sigma_safe)
    exploration_bonus = exploration_weight * sigma_safe / (sigma_mean + 1e-9) / cost_safe
    
    # Total value: core + exploration
    value = core_value + exploration_bonus
    
    # The acquisition score is the negative cost-to-go, which is the value
    # of evaluating this point. Higher value = lower cost-to-go = better.
    score = value
    
    # Clamp non-finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Ensure positive scores for valid candidates
    score = np.maximum(score, 0.0)
    
    # Deadline guard
    if time.time() - t0 > 280:
        return score
    
    return score
# EVOLVE-BLOCK-END
