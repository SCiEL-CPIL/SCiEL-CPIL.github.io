"""Cost-aware acquisition function using Thompson Sampling per unit cost."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    We sample a single function value from the GP posterior for each candidate
    and select the candidate that minimizes the ratio of the sampled value to
    the evaluation cost. This naturally balances stochastic trade-offs between
    predicted objective improvement and acquisition cost.
    
    For minimization, we want to maximize the score, so we return:
        score = -sampled_value / cost
    """
    start_time = time.time()
    deadline = 290.0  # Safety margin for 300s limit
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases: avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Thompson Sampling: draw one sample from N(mu, sigma^2) for each candidate
    # Using standard normal random numbers
    eps = np.random.standard_normal(size=mu.shape)
    samples = mu + sigma_safe * eps
    
    # Score for minimization: we want to maximize -sampled_value / cost
    # Lower sampled values are better, and lower cost is better
    score = -samples / cost_safe
    
    # Check deadline
    if time.time() > deadline:
        # Fallback: return deterministic EI-based scores if time is running out
        # But since we've already computed samples, just return what we have
        pass
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(score), score, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
