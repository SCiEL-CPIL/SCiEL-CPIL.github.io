# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB) per unit cost.
    
    UCB provides an explicit exploration-exploitation trade-off via the beta parameter.
    For minimization, we use the lower confidence bound (LCB) as the optimistic estimate:
    LCB(x) = mu(x) - beta * sigma(x)
    
    We want to minimize LCB(x). Thus, we maximize -LCB(x).
    To account for cost, we divide by the cost of evaluation.
    
    Score = (-mu + beta * sigma) / cost
    """
    # Avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    # Exploration parameter
    # A value of 2.0 is a common starting point for BO, balancing exploration and exploitation.
    beta = 2.0
    
    # Optimistic lower bound for minimization
    # We want to find x that minimizes f(x).
    # The most optimistic value we can expect is mu - beta*sigma.
    # We want to maximize the "potential" which is the negative of this bound.
    score = (-mu + beta * sigma) / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
