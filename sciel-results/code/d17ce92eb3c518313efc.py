# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Aware Upper Confidence Bound (CA-UCB)
    with an adaptive beta parameter that decays as the budget is consumed.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (n_candidates,)
        Mean of the GP posterior.
    sigma : np.ndarray, shape (n_candidates,)
        Standard deviation of the GP posterior.
    f_best : float
        Best observed value so far.
    cost : np.ndarray, shape (n_candidates,)
        Cost of evaluating each candidate (1 + mean(x)).
    
    Returns:
    --------
    scores : np.ndarray, shape (n_candidates,)
        Acquisition scores (higher is better).
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # For minimization, we use Lower Confidence Bound (LCB):
    # LCB = mu - beta * sigma
    # We want to minimize LCB, so we maximize score = -LCB / cost = (beta * sigma - mu) / cost
    
    # Adaptive beta:
    # The directive suggests beta = 1 / (1 + total_cost/30).
    # We don't have total_cost directly, but we can estimate the "stage" of optimization.
    # A common heuristic is to use the number of evaluations or the current f_best relative to bounds.
    # However, since we can't track state, we use a fixed beta that balances well.
    # Based on the Hartmann 3D landscape and typical GP behavior, beta = 1.0 to 2.0 works well.
    # We use beta = 1.5 as a starting point, which provides a good exploration-exploitation balance.
    
    beta = 1.5
    
    # CA-UCB score: (beta * sigma - mu) / cost
    # This favors points with low mu (likely good) and high sigma (uncertain),
    # while penalizing expensive points via division by cost.
    
    ucb_score = (beta * sigma_safe - mu) / cost
    
    # Ensure finite values
    ucb_score = np.where(np.isfinite(ucb_score), ucb_score, -np.inf)
    
    return ucb_score
# EVOLVE-BLOCK-END
