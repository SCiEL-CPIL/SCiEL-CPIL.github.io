"""Rewritten program using Cost-Adjusted Upper Confidence Bound (UCB)."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Adjusted Upper Confidence Bound (UCB).
    
    This approach uses the UCB strategy, which balances exploitation (low predicted
    mean) and exploration (high predicted uncertainty). For minimization, we
    maximize the score: (-mu + kappa * sigma) / cost.
    
    This is distinct from Expected Improvement methods as it does not depend
    explicitly on f_best in the acquisition formula, making it robust to
    early-stage noise in the best observed value.
    """
    # Exploration parameter. 1.5 is a common heuristic for small domains.
    kappa = 1.5
    
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 0.0)
    
    # UCB score for minimization:
    # We want low mu (good) and high sigma (uncertain).
    # Maximizing -mu + kappa * sigma achieves this.
    ucb_score = -mu + kappa * sigma_safe
    
    # Cost-aware weighting
    # Divide by cost to penalize expensive evaluations.
    score = ucb_score / cost
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
