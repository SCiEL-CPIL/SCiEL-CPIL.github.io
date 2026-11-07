# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using GP-UCB with cosine-annealed exploration.
    Uses f_best to estimate progress and decays beta using a cosine schedule.
    Aggressively penalizes high-cost candidates via cost normalization.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero in cost
    cost_safe = np.maximum(cost, 1e-9)
    
    # Estimate progress based on f_best.
    # Increased f_ref to 50.0 to better match typical initial random evaluations
    # for the Three Hump Camel function in [-5, 5]^2.
    f_ref = 50.0
    # Clip progress to [0, 1]
    progress = 1.0 - (f_best / f_ref)
    progress = np.clip(progress, 0.0, 1.0)
    
    # Cosine annealing for beta decay
    # Starts at beta_max, ends at beta_min, smooth transition
    beta_max = 4.0
    beta_min = 0.2
    beta = beta_min + 0.5 * (beta_max - beta_min) * (1.0 + np.cos(np.pi * progress))
    
    # UCB for minimization: maximize -mu + beta * sigma
    ucb_score = -mu + beta * sigma
    
    # Cost normalization: divide by convex combination of cost and constant
    # Increased alpha to 0.8 to more strongly penalize high-cost candidates
    alpha = 0.8
    c_const = 1.0
    denom = alpha * cost_safe + (1 - alpha) * c_const
    denom = np.maximum(denom, 1e-9)
    
    scores = ucb_score / denom
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
