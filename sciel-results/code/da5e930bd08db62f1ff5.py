# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function: Upper Confidence Bound (UCB) for minimization,
    normalized by evaluation cost.
    
    UCB encourages exploration by adding a confidence term (beta * sigma) to the
    predicted mean. For minimization, we look for low values, so we maximize
    (-mu + beta * sigma). Dividing by cost prioritizes points that offer high
    potential improvement per unit cost.
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # UCB parameter: controls the trade-off between exploitation and exploration.
    # A value of 1.5 is a common heuristic for multimodal functions like Rastrigin.
    beta = 1.5

    # UCB score for minimization:
    # We want to find x that minimizes f(x).
    # The lower bound of f(x) is approx mu - beta * sigma.
    # We want to maximize the "potential goodness", which is - (lower bound) = -mu + beta * sigma.
    ucb_score = -mu + beta * sigma_safe

    # Cost-aware score
    scores = ucb_score / cost_safe

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
