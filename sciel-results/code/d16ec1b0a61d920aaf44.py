"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_ACQ_START_TIME = time.time()
_ACQ_DEADLINE = _ACQ_START_TIME + 280
_ACQ_CALL_COUNT = [0]


def acquisition(mu, np_sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of rosenbrock 2d.
    
    Parameters:
    - mu: array of shape (2000,), GP mean predictions
    - np_sigma: array of shape (2000,), GP standard deviation predictions
    - f_best: scalar, best observed value so far
    - cost: array of shape (2000,), evaluation costs for each candidate
    
    Returns:
    - array of shape (2000,), acquisition scores (higher = better)
    """
    _ACQ_CALL_COUNT[0] += 1

    # Deadline guard
    if time.time() > _ACQ_DEADLINE:
        # Fallback: simple cost-aware exploitation
        fallback = -mu / np.maximum(cost, 1e-9)
        return np.where(np.isfinite(fallback), fallback, 0.0)

    # Sanitize inputs
    sigma = np.maximum(np_sigma, 1e-12)
    cost = np.maximum(cost, 1e-9)

    # Thompson Sampling:
    # Sample a single latent function value for each candidate from the GP predictive distribution.
    # For minimization, we want to find low values. 
    # The predictive distribution is N(mu, sigma^2).
    # We sample y ~ N(mu, sigma^2).
    # To make the score "higher is better", we can maximize -y / cost.
    # This effectively samples a "regret" or "improvement" proxy.
    # If y is very low (good), -y is high (good score).
    # If cost is high, the score is penalized.
    
    # Generate standard normal noise
    # Use a fixed seed per call? No, Thompson sampling relies on randomness for exploration.
    # However, we must ensure reproducibility isn't required for the *scores* themselves, 
    # but the *selection* is stochastic.
    # The problem states "Return 2000 finite scores". The evaluator picks the argmax.
    # So we just need to return the sampled scores.
    
    # One potential issue: if we just sample once, the variance of the scores is high.
    # But that's the point of Thompson Sampling.
    
    # Let's refine the sampling.
    # y = mu + sigma * z, where z ~ N(0, 1)
    z = np.random.standard_normal(len(mu))
    sampled_values = mu + sigma * z
    
    # Score: We want to maximize the "quality" per cost.
    # Quality of a point x is inversely related to its function value f(x).
    # So score = -sampled_values / cost.
    
    scores = -sampled_values / cost
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
