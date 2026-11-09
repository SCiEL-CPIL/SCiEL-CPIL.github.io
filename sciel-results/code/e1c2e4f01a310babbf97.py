import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on hartmann 6d.
    Uses Thompson Sampling: draw a single sample from the posterior for all candidates,
    then select the point that minimizes y/cost (i.e., maximize -y/cost).
    
    Args:
        mu: Mean of the GP prediction, shape (N,)
        sigma: Standard deviation of the GP prediction, shape (N,)
        f_best: Best observed value so far (scalar).
        cost: Cost of evaluating each point, shape (N,)
        
    Returns:
        scores: Acquisition scores, shape (N,), higher = evaluate next
    """
    start_time = time.time()
    DEADLINE = 325.0  # Safety margin before 330s kill
    
    # Input validation and safety
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure shapes are correct
    n = mu.shape[0]
    
    # Handle non-finite values
    sigma_safe = np.where(np.isfinite(sigma), sigma, 1e-8)
    sigma_safe = np.maximum(sigma_safe, 1e-8)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    cost_safe = np.where(np.isfinite(cost), cost, 1.0)
    cost_safe = np.maximum(cost_safe, 1e-8)
    
    # Thompson Sampling: draw a single random sample from the posterior
    # y ~ N(mu, sigma^2) for each candidate
    # We draw a single random scalar z and use y = mu + sigma * z
    # This creates a correlated sample across all candidates, which is a valid
    # way to perform Thompson Sampling with a single GP posterior sample.
    z = np.random.randn()
    
    # Sampled values from the posterior
    y_sampled = mu + sigma_safe * z
    
    # Cost-aware score: we want to minimize y_sampled / cost_safe
    # So the acquisition score (higher is better) is -y_sampled / cost_safe
    scores = -y_sampled / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Deadline check
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Return a safe fallback
        return np.zeros(n, dtype=np.float64)
    
    return scores
