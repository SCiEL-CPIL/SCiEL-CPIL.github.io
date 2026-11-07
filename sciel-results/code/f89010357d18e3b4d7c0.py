"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a Thompson Sampling heuristic.
    
    Parameters:
    mu: np.ndarray of shape (N,) - Mean of the GP at candidate points.
    sigma: np.ndarray of shape (N,) - Standard deviation of the GP at candidate points.
    f_best: float - The best (lowest) observed objective value so far.
    cost: np.ndarray of shape (N,) - Cost of evaluating each candidate.
    
    Returns:
    scores: np.ndarray of shape (N,) - Acquisition scores (higher is better).
    """
    # Thompson Sampling heuristic:
    # Instead of computing a deterministic bound like LCB (mu - beta * sigma),
    # we sample a plausible objective value from the posterior predictive distribution.
    # Since we cannot easily maintain state between calls for a single sample per candidate
    # in a vectorized acquisition function that returns scores for 2000 candidates,
    # we use a single global random draw z ~ N(0, 1) and apply it to all candidates.
    # This effectively creates a random linear combination of mu and sigma for this step.
    # score_i = -(mu_i + z * sigma_i) / cost_i
    # Maximizing this score is equivalent to minimizing the sampled value per unit cost.
    
    # Generate a single standard normal random number for this iteration.
    # This introduces stochasticity that changes with each call, allowing the GP
    # to explore different regions based on the current uncertainty landscape.
    z = np.random.normal(0, 1)
    
    # Calculate the sampled objective values
    # We want to minimize f, so the "sampled value" is mu + z * sigma.
    # If z is positive, we favor low sigma (exploitation).
    # If z is negative, we favor high sigma (exploration).
    sampled_values = mu + z * sigma
    
    # We want to select the point with the lowest sampled value per unit cost.
    # Therefore, the score to maximize is -sampled_value / cost.
    safe_cost = np.maximum(cost, 1e-9)
    scores = -sampled_values / safe_cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
