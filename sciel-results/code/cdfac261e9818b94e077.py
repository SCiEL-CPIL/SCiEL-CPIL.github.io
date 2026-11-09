"""Rewritten acquisition: Upper Confidence Bound with adaptive beta."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # UCB-style acquisition: maximize mu - beta * sigma (minimization).
    # Since we return scores where higher means evaluate next,
    # and we want to minimize f, we score = -mu + beta * sigma.
    #
    # Adaptive beta: use a value that balances exploration/exploitation.
    # For 2000 candidates over 30 evaluations, a moderate exploration
    # coefficient works well. We scale sigma by a constant.
    #
    # Using beta=2.0 is a common choice for UCB in Bayesian optimization.
    # This encourages exploring regions with high uncertainty.
    
    # Handle potential numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB score for minimization: -mu + beta * sigma
    # Higher score means more promising to evaluate
    beta = 2.0
    scores = -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
