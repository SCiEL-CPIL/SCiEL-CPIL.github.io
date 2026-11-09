"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB).
    
    Instead of Expected Improvement (EI), this uses a UCB strategy which
    balances exploitation (low mean) and exploration (high uncertainty)
    linearly. The score is normalized by the evaluation cost to prefer
    cheaper evaluations.
    
    Score = -(mu + beta * sigma) / cost
    """
    beta = 3.0
    
    # Calculate UCB for minimization (we want to minimize f, so we look at
    # the upper bound of the posterior distribution of f(x))
    # We want to pick x that minimizes mu + beta * sigma.
    # To maximize the score, we negate this.
    ucb = mu + beta * sigma
    
    # Normalize by cost. Lower cost is better, so dividing by cost rewards
    # low-cost points.
    # Add epsilon to cost to avoid division by zero (though cost >= 1).
    score = -ucb / (cost + 1e-10)
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
