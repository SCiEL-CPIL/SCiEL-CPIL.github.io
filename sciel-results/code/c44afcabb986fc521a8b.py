# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using GP-UCB with progress-scaled exploration.
    Uses f_best to estimate progress and decay beta linearly.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero in cost
    cost_safe = np.maximum(cost, 1e-9)
    
    # Estimate progress based on f_best.
    # The minimum of the Three Hump Camel is 0.
    # Use a lower reference value to maintain higher exploration for longer,
    # helping escape local minima in the rugged landscape.
    f_ref = 5.0
    # Clip progress to [0, 1]
    progress = 1.0 - (f_best / f_ref)
    progress = np.clip(progress, 0.0, 1.0)
    
    # Beta decays linearly from beta_max to beta_min as progress goes from 0 to 1
    # Increased beta_max for stronger initial exploration.
    beta_max = 4.5
    beta_min = 0.5
    beta = beta_max - progress * (beta_max - beta_min)
    
    # UCB for minimization: maximize -mu + beta * sigma
    ucb_score = -mu + beta * sigma
    
    # Cost normalization: Convex combination of cost and constant.
    # Higher alpha places more weight on cost, penalizing expensive regions more strongly.
    alpha = 0.8
    c_const = 1.0
    denom = alpha * cost_safe + (1 - alpha) * c_const
    denom = np.maximum(denom, 1e-9)
    
    scores = ucb_score / denom
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
