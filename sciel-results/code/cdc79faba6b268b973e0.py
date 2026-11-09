# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Aware Upper Confidence Bound (UCB) for minimization.
    
    The acquisition function is defined as:
    Score(x) = (xi * sigma(x) - mu(x)) / cost(x)
    
    This approach is distinct from Expected Improvement (EI) based methods.
    UCB balances exploitation (low mu) and exploration (high sigma) by
    estimating a lower bound on the function value: mu(x) - xi * sigma(x).
    We want to minimize this lower bound, which is equivalent to maximizing
    its negative, normalized by the evaluation cost.
    
    Parameters:
    mu: Mean predictions from the GP (shape 2000,)
    sigma: Standard deviation predictions from the GP (shape 2000,)
    f_best: Best observed function value (scalar)
    cost: Cost of evaluating each candidate (shape 2000,)
    
    Returns:
    scores: Acquisition scores (shape 2000,)
    """
    # Hyperparameter for exploration-exploitation trade-off
    # A higher value encourages more exploration.
    # For multimodal problems like Styblinski-Tang, a value around 1.5-2.0 is typical.
    xi = 1.8
    
    # Safe division
    cost_safe = cost + 1e-9
    
    # Calculate UCB score
    # We minimize mu - xi*sigma, so we maximize -(mu - xi*sigma) = xi*sigma - mu
    scores = (xi * sigma - mu) / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
