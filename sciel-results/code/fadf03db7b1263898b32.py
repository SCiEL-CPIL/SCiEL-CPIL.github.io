# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Aware Upper Confidence Bound (cUCB) acquisition function.
    
    This approach uses the Upper Confidence Bound (UCB) to balance exploitation
    (low mean) and exploration (high uncertainty). For minimization, we maximize
    -mu + kappa * sigma. To account for evaluation costs, we divide the UCB score
    by the cost of the candidate point.
    
    Steps:
    1. Calculate the UCB score: ucb = -mu + kappa * sigma.
    2. Divide by cost to get cost-aware score.
    3. Ensure finite values.
    
    Note: Unlike EI, UCB does not depend directly on f_best in the calculation,
    relying instead on the GP posterior (mu, sigma) which is updated as new
    points are evaluated.
    """
    # Exploration parameter
    kappa = 1.0
    
    # Ensure sigma is non-negative and cost is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # UCB for minimization: we want low mu and high sigma.
    # Maximizing -mu + kappa * sigma achieves this.
    ucb = -mu + kappa * sigma_safe
    
    # Cost-adjusted UCB
    score = ucb / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
