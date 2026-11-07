"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# We keep a simple deadline guard.
# The evaluator kills after 330s. We aim to return well before that.
# The acquisition function itself is vectorized and fast, but we add a guard just in case.
# Since the function is called repeatedly in a loop, we can't easily track total elapsed time
# across calls without global state, which is fragile. However, the task says "Each evaluation
# must finish within 300 s." The loop runs while cost < 30.
# The main cost is in the BO loop outside this function.
# This function just computes scores for 2000 points. It should be very fast (<0.1s).
# So a deadline guard here is likely unnecessary but we can add a simple check if inputs are huge.
# Actually, the inputs are always shape (2000,). So it is fixed size.
# We will focus on improving the acquisition function itself.

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization using Expected Improvement (EI).
    
    Args:
        mu: (2000,) array of GP mean predictions.
        sigma: (2000,) array of GP standard deviation predictions.
        f_best: float, the lowest observed function value.
        cost: (2000,) array of costs for evaluating each point.
    
    Returns:
        (2000,) array of scores (higher is better).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero and invalid probabilities
    sigma = np.maximum(sigma, 1e-9)
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Expected Improvement (EI) for minimization:
    # We want to maximize the expected improvement over f_best.
    # Improvement = max(f_best - f(x), 0)
    # E[Improvement] = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Z = (f_best - mu) / sigma
    
    # Calculate Z
    z = (f_best - mu) / sigma
    
    # Calculate Phi(Z) and phi(Z)
    phi_z = norm.pdf(z)
    phi_cdf_z = norm.cdf(z)
    
    # Calculate Expected Improvement
    ei = (f_best - mu) * phi_cdf_z + sigma * phi_z
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score:
    # We divide by cost to prioritize cheaper evaluations.
    # Using a power of cost > 1 can penalize expensive points more heavily, 
    # but a simple division is a standard baseline. 
    # Given the cost is 1 + mean(x), it ranges from 1 to 2.
    # Let's use a simple division.
    
    score = ei / cost
    
    # Ensure scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
