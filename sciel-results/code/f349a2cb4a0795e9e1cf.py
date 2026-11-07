"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def _expected_improvement(mu, sigma, f_best):
    """
    Compute Expected Improvement (EI) for minimization.
    E[max(f_best - f(x), 0)] where f(x) ~ N(mu, sigma^2)
    """
    # Ensure sigma is positive to avoid division by zero
    s = np.maximum(sigma, 1e-8)
    
    # Standard normal variable
    z = (f_best - mu) / s
    
    # EI formula
    # E[I] = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + s * norm.pdf(z)
    
    # EI is non-negative by definition, but numerical errors might cause tiny negatives
    ei = np.maximum(ei, 0.0)
    
    # Handle non-finite values
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Args:
        mu: array of shape (N,) - GP mean predictions
        sigma: array of shape (N,) - GP standard deviation predictions
        f_best: float - Best function value observed so far
        cost: array of shape (N,) - Cost of evaluating each candidate
    
    Returns:
        scores: array of shape (N,) - Acquisition scores (higher is better)
    """
    t_start = time.time()
    
    # Ensure inputs are arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # 1. Calculate Expected Improvement
    ei = _expected_improvement(mu, sigma, f_best)
    
    # 2. Apply Cost-Awareness
    # The cost is given as 1 + mean(x_scaled). 
    # We want to maximize improvement per unit cost.
    # Dividing by cost effectively penalizes expensive points.
    # Using a power of 1.0 is a standard "cost-effectiveness" metric.
    # We can adjust the exponent if we want to be more or less aggressive with cost.
    # Let's stick to 1.0 for a direct "improvement per cost" interpretation.
    
    safe_cost = np.maximum(cost, 1e-8)
    
    # Score = EI / cost
    scores = ei / safe_cost
    
    # 3. Safety Checks
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Optional: Add a tiny epsilon to avoid ties? Not strictly necessary.
    
    # 4. Deadline Guard
    # The problem states the evaluator kills the program after 330s.
    # This function is called in a loop. We should ensure it returns quickly.
    # If this function takes too long, it might contribute to timeout.
    # However, the operations here are O(N) with N=2000, which is very fast (<1ms).
    # No complex loops or heavy sampling are used, so timeout risk is low.
    
    elapsed = time.time() - t_start
    if elapsed > 280: # Should never happen for this simple calculation
         return scores
         
    return scores
# EVOLVE-BLOCK-END
