import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Adjusted UCB.
    
    Args:
    mu: array of shape (n_candidates,), predicted mean in objective units.
    sigma: array of shape (n_candidates,), predicted std in objective units.
    f_best: float, best observed value.
    cost: array of shape (n_candidates,), cost of evaluating each candidate.
    
    Returns:
    scores: array of shape (n_candidates,), higher is better.
    """
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-9)
    
    # Cost-Adjusted UCB
    # We want to minimize f.
    # Optimistic estimate of minimum: -mu + beta * sigma
    # Divide by cost to penalize expensive evaluations.
    
    # Beta tuning:
    # Given the small budget (~5-6 evals), high exploration is needed.
    # Beta = 2.5 to 3.0 is a reasonable heuristic for this scale.
    beta = 3.0
    
    # Calculate UCB scores
    ucb_scores = -mu + beta * sigma
    
    # Normalize by cost
    # cost is typically in [2, 4]
    scores = ucb_scores / cost
    
    return scores
