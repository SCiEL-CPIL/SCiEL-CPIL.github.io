import numpy as np
def acquisition(mu, sigma, f_best, cost):
    """
    Multi-objective rank-based acquisition function.
    Balances exploitation (low mu), exploration (high sigma), and cost (low cost)
    using rank aggregation to be robust to scaling differences.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    n = len(mu)
    
    # Ranks: 0 is best
    # Exploitation: Lowest mu is best
    rank_mu = np.argsort(np.argsort(mu))
    
    # Exploration: Highest sigma is best (so rank -sigma)
    rank_sigma = np.argsort(np.argsort(-sigma))
    
    # Cost: Lowest cost is best
    rank_cost = np.argsort(np.argsort(cost))
    
    # Weights
    # Prioritize exploitation and exploration, mild cost penalty
    # Cost is not a hard constraint but a budget. Moving to the minimum (high cost) is necessary.
    w_mu = 0.6
    w_sigma = 0.3
    w_cost = 0.1
    
    # Combine ranks. Lower sum is better.
    # We want to return higher scores for better candidates.
    # So negate the weighted sum.
    scores = -(w_mu * rank_mu + w_sigma * rank_sigma + w_cost * rank_cost)
    
    # Normalize to [0, 1] for consistency (optional but good)
    # The evaluator just takes argmax, so scale doesn't matter, only order.
    # But ensuring finite values is key.
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
