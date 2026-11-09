"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0  # 5s safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - lowest observed function value
        cost: array of shape (2000,) - cost of evaluating each candidate
        
    Returns:
        array of shape (2000,) - acquisition scores (higher is better)
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a valid finite array to avoid invalidating the result
        # If we run out of time, we return a random but finite score
        return np.random.uniform(size=mu.shape)
    
    # Ensure sigma is positive for numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Thompson Sampling: sample from the posterior predictive distribution
    # For minimization, we want to find low values.
    # We sample a value y ~ N(mu, sigma^2) for each candidate.
    # Then we select the candidate with the lowest y, adjusted for cost.
    
    # Generate random samples
    # Using standard normal distribution
    z = np.random.standard_normal(mu.shape)
    
    # Sampled values: y = mu + sigma * z
    sampled_values = mu + sigma_safe * z
    
    # To make this cost-aware, we can either:
    # 1. Minimize sampled_value / cost (if cost is a multiplier on regret)
    # 2. Minimize sampled_value - lambda * cost (if cost is additive penalty)
    # The task says "Evaluating x costs 1 + mean(x scaled to [0, 1])".
    # Usually in cost-aware BO, we maximize the expected improvement per unit cost,
    # or minimize the expected regret per unit cost.
    # Here, we approximate by dividing the sampled value by the cost.
    # Lower is better for the objective, so we want low sampled_values / cost.
    # Since the acquisition function returns scores where HIGHER is better to evaluate next,
    # we should return - (sampled_values / cost) or similar.
    
    cost_safe = np.maximum(cost, 1e-8)
    
    # Score: we want to maximize the "desirability".
    # Desirability is high when sampled_value is low and cost is low.
    # So score = - sampled_value / cost
    scores = - sampled_values / cost_safe
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
