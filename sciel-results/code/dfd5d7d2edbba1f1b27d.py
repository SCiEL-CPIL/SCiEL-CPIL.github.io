# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB).
    
    This approach balances exploration and exploitation by considering the
    lower bound of the predicted function value (mu - beta * sigma) and
    normalizing the expected improvement by the evaluation cost.
    
    UCB is often more effective than EI in early stages of optimization
    with small budgets because it explicitly encourages exploring regions
    with high uncertainty.
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB parameter: balances exploration and exploitation
    # A value of 2.0 is a standard heuristic for small to medium budgets
    beta = 2.0
    
    # Calculate the UCB lower bound (for minimization)
    # We want to find points where the lower bound is significantly lower than f_best
    ucb_lower_bound = mu - beta * sigma_safe
    
    # Expected improvement based on UCB
    # We only care about improvements over f_best
    # If the UCB lower bound is better than f_best, we have a potential improvement
    improvement = np.maximum(0.0, f_best - ucb_lower_bound)
    
    # Normalize by cost to prefer cheaper evaluations
    # cost is expected to be positive
    cost_safe = np.maximum(cost, 1e-9)
    
    score = improvement / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
