# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Upper Confidence Bound (UCB) divided by cost.
    This approach uses the probabilistic upper confidence bound of the GP
    posterior. It balances exploration and exploitation by maximizing the
    upper bound of the function value (adjusted for minimization) per unit
    of evaluation cost.
    """
    # Beta controls the exploration-exploitation trade-off
    # A higher beta encourages more exploration
    beta = 2.5
    
    # Calculate UCB for minimization:
    # We want to find points with low mu and high sigma.
    # Maximizing (-mu + beta * sigma) achieves this.
    ucb = -mu + beta * sigma
    
    # Cost-aware score: UCB per unit cost
    # Higher score is better
    score = ucb / cost
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
