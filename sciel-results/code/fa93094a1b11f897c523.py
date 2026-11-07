# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Weighted UCB (Upper Confidence Bound).
    
    For minimization, we define the Lower Confidence Bound (LCB) as:
    LCB(x) = mu(x) - beta * sigma(x)
    
    We want to select points with low LCB and low cost.
    Therefore, we maximize the score:
    Score(x) = (f_best - LCB(x)) / cost(x)
             = (f_best - mu(x) + beta * sigma(x)) / cost(x)
             
    This linear combination balances exploitation (f_best - mu) and exploration (beta * sigma),
    normalized by the evaluation cost.
    """
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Cost safe division
    cost_safe = np.maximum(cost, 1e-9)
    
    # Exploration parameter beta
    # Tuned for multimodal functions (Ackley, Rastrigin, Levy, Hartmann)
    # Higher beta encourages more exploration
    beta = 2.0
    
    # Calculate the UCB-based score
    # We use f_best as a reference to ensure that points currently better than f_best
    # get a high score, and points worse get lower scores.
    # The term (f_best - mu) favors exploitation.
    # The term (beta * sigma) favors exploration.
    numerator = f_best - mu + beta * sigma_safe
    
    # Normalize by cost
    scores = numerator / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
