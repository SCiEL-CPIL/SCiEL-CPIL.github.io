"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function for minimization.
    We maximize the score: mu - sqrt(beta) * sigma.
    A higher beta encourages more exploration.
    """
    # Use a beta value that balances exploration and exploitation well for small budgets.
    # Beta = 2.0 is a common starting point for small N.
    beta = 2.0
    sqrt_beta = np.sqrt(beta)
    
    # Calculate UCB score. Since we are minimizing f(x), we want to find regions 
    # where f(x) could be very low. The lower bound of the confidence interval 
    # is mu - sqrt(beta)*sigma. Maximizing this score targets regions with 
    # potentially low function values.
    scores = mu - sqrt_beta * sigma
    
    # Ensure all scores are finite to satisfy the contract.
    # If sigma is huge or mu is weird, handle NaNs/Infs.
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
