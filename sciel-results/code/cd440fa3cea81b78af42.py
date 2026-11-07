# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Lower Confidence Bound (LCB).
    
    Unlike Expected Improvement (EI), LCB is an optimistic strategy that
    balances exploitation (low mu) and exploration (high sigma) linearly.
    The score is the negated LCB normalized by the evaluation cost,
    favoring points that are likely to be low-cost and have a low
    predicted function value.
    """
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Exploration parameter for LCB
    # A value of 2.0 is a common heuristic for balancing exploration/exploitation
    beta = 2.0
    
    # Lower Confidence Bound: we want to minimize this
    # lcb = mu - beta * sigma
    # We want to MAXIMIZE the acquisition score, so we use -lcb
    # score_unnormalized = -(mu - beta * sigma) = beta * sigma - mu
    
    score_unnormalized = beta * sigma_safe - mu
    
    # Normalize by cost to prefer cheaper evaluations
    # Cost is in the range [1, 2]
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = score_unnormalized / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
