import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Upper Confidence Bound (UCB) acquisition function.
    
    Parameters:
    mu : np.ndarray, shape (2000,)
        Mean of the Gaussian Process predictions.
    sigma : np.ndarray, shape (2000,)
        Standard deviation of the Gaussian Process predictions.
    f_best : float
        The best observed function value so far (minimum).
    cost : np.ndarray, shape (2000,)
        The cost of evaluating each candidate point.
        
    Returns:
    scores : np.ndarray, shape (2000,)
        Acquisition scores for each candidate. Higher is better.
    """
    # Exploration-exploitation trade-off parameter
    # Based on measured knowledge, beta=0.8 outperforms beta=1.0 for this task
    beta = 0.8
    
    # For minimization, the optimistic lower bound is mu - beta * sigma
    optimistic_values = mu - beta * sigma
    
    # Calculate the potential improvement over the current best
    # We only care about points that are potentially better than f_best
    improvements = f_best - optimistic_values
    
    # Clip negative improvements to 0 (no acquisition value if optimistic bound is worse than best)
    improvements = np.maximum(0, improvements)
    
    # Normalize by cost to get cost-aware score
    # Add small epsilon to avoid division by zero
    safe_cost = cost + 1e-9
    scores = improvements / safe_cost
    
    return scores
