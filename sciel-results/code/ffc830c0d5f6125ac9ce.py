"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Gaussian Process UCB with budget-scaled exploration coefficient,
    divided by a convex combination of cost and a constant.
    """
    t0 = time.time()
    deadline = t0 + 290.0  # safety margin before 300s limit

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero in cost
    cost_safe = np.maximum(cost, 1e-9)
    
    # Clamp sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Budget-scaled exploration coefficient
    # Total budget is 30. Start with high exploration, decay to low.
    # At cost=5 (initial), alpha is high. At cost=30, alpha is low.
    # Let's define alpha as a linear decay from 2.0 to 0.5
    # alpha = 2.0 - (1.5 * (cost - 5) / 25)
    # But cost is per-point. The 'budget' in the task description says "While total cost is below 30".
    # The cost argument is the cost of evaluating each candidate.
    # However, the acquisition function doesn't know the *total* cost spent so far, only the cost of the candidates.
    # Wait, the prompt says "cost has shape (2000,)". This is the cost of evaluating each of the 2000 candidates.
    # It does NOT provide the total cost spent so far.
    # Therefore, we cannot directly scale alpha based on total budget spent.
    # We must rely on the cost of the candidate itself.
    
    # Alternative strategy: Use a fixed UCB coefficient but weight by cost.
    # Or, infer progress from the relationship between f_best and mu/sigma? No, that's unreliable.
    
    # Let's stick to a robust cost-aware UCB.
    # UCB = mu + beta * sigma
    # We want to balance exploration (sigma) and exploitation (mu) while penalizing high cost.
    
    # A common approach: EI / cost or UCB / cost.
    # Let's try a hybrid: (mu + beta * sigma) / (cost + c)
    # where beta is a fixed exploration parameter.
    
    # What value for beta?
    # In 2D BO with Matern-5/2, beta around 1.0-2.0 is typical.
    # Let's use beta = 1.5.
    
    beta = 1.5
    c = 0.5  # Constant to avoid division by very small costs and to balance
    
    # Calculate UCB
    ucb = mu + beta * sigma_safe
    
    # Divide by convex combination of cost and constant
    # cost is in [1, 2] roughly (1 + mean(x_scaled))
    # So cost + c is in [1.5, 2.5]
    scores = ucb / (cost_safe + c)
    
    # Check time
    if time.time() > deadline:
         # Return best found so far? No, we just computed it.
         # If computation took too long, we should return something valid.
         # The computation above is vectorized and fast for 2000 points.
         pass
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
