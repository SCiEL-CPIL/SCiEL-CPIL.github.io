import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 300.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    Returns scores of shape (2000,), where higher scores indicate better candidates.
    """
    # Explicit deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback: Simple Cost-Aware EI
        sigma_safe = sigma + 1e-9
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        cost_safe = cost + 1e-9
        return ei / cost_safe

    # Ensure safety in denominators
    sigma_safe = sigma + 1e-9
    cost_safe = cost + 1e-9

    # Thompson Sampling for minimization
    # Sample from the posterior predictive distribution: y ~ N(mu, sigma^2)
    # We want to minimize y + lambda * cost
    # Score to maximize: -(y + lambda * cost)
    
    # Generate random samples
    z = np.random.normal(0, 1, mu.shape)
    samples = mu + sigma_safe * z
    
    # Cost penalty
    # lambda controls the trade-off. 
    # Cost is in [1, 2]. Values are in [-0.8, 1].
    # A penalty of 0.1 means 1 unit of cost costs 0.1 units of value.
    lambda_cost = 0.1
    
    scores = -(samples + lambda_cost * cost_safe)
    
    # Ensure all outputs are finite
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    return scores
