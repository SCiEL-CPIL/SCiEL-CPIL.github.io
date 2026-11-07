import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    Uses a fixed pseudo-random vector for the posterior sample to ensure
    deterministic behavior (reproducibility) while maintaining the
    stochastic structure of Thompson Sampling.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Generate a fixed set of standard normal samples for Thompson Sampling
    # Using a fixed seed ensures the same 'z' vector is used for every call,
    # making the acquisition deterministic.
    rng = np.random.default_rng(42)
    z = rng.standard_normal(len(mu))
    
    # Sample from posterior predictive: theta = mu + sigma * z
    theta = mu + sigma_safe * z
    
    # Normalized regret: (theta - f_best) / sqrt(cost)
    # We want to minimize regret, so we maximize the negative regret.
    # Score = (f_best - theta) / sqrt(cost)
    scores = (f_best - theta) / np.sqrt(cost_safe)
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
