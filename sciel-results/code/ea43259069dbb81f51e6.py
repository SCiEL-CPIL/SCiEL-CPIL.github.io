# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Thompson sampling based acquisition with cost-aware exploration.
    Samples from the posterior and selects candidates maximizing
    the sampled minimum per unit cost, with adaptive exploration.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero or negative values
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Sample from the posterior: theta = mu + sigma * z, where z ~ N(0,1)
    # This is a single Thompson sampling draw
    z = np.random.standard_normal(mu.shape)
    sampled_values = mu + sigma_safe * z
    
    # We want to minimize the function, so we want low sampled values.
    # The acquisition score should be high for candidates we want to evaluate.
    # So we take negative of the sampled value (since lower is better for the objective).
    # Then divide by cost to get cost-aware score.
    
    # Add a small epsilon to cost to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    # Base score: negative sampled value divided by cost
    base_scores = -sampled_values / cost_safe
    
    # Add exploration bonus for high uncertainty regions
    # This encourages exploration in areas where the GP is uncertain
    exploration_bonus = sigma_safe / cost_safe * 0.5
    
    scores = base_scores + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Normalize scores to prevent numerical issues
    # Center and scale so that the best score is around 0
    score_max = np.max(scores)
    if np.isfinite(score_max) and score_max > 0:
        scores = scores - score_max
    
    return scores
# EVOLVE-BLOCK-END
