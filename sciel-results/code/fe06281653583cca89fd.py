"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for Thompson Sampling to ensure reproducibility within a single run if needed, 
# but since we are called 2000 times in a batch, we just need one random vector per call.
# We don't need persistent state between calls if we treat each call as a fresh "round" of sampling.
# However, to be safe and fast, we just generate random numbers.

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Parameters:
    mu: array of shape (N,) mean of GP posterior
    sigma: array of shape (N,) std of GP posterior
    f_best: float, best observed value so far
    cost: array of shape (N,) cost of evaluating each candidate
    
    Returns:
    scores: array of shape (N,) acquisition scores (higher is better)
    """
    # Generate samples from the posterior predictive distribution
    # y_i ~ N(mu_i, sigma_i)
    # We use a single random vector to keep the computation efficient and consistent across the batch
    z = np.random.randn(len(mu))
    samples = mu + sigma * z
    
    # Calculate the "improvement" of each sample relative to f_best
    # Improvement is positive if the sample is better (lower) than f_best
    improvements = f_best - samples
    
    # Only positive improvements are relevant for the "best" selection in TS context
    # If sample is worse than f_best, improvement is negative (or 0 if we clamp).
    # In standard TS for minimization, we just pick min(sample).
    # But we need cost awareness.
    # Metric: Improvement per unit cost.
    # If improvement is negative, the score should be 0 or negative?
    # The task says "higher = evaluate next".
    # If a sample is worse than f_best, it has 0 expected improvement.
    # So we clamp improvements to 0.
    improvements = np.maximum(improvements, 0.0)
    
    # Avoid division by zero or very small costs
    safe_cost = np.maximum(cost, 1e-9)
    
    scores = improvements / safe_cost
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
