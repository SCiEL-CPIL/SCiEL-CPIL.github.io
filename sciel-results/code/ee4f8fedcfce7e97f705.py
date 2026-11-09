import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_LIMIT = 285.0  # Safety margin below 300s

def acquisition(mu, sigma, f_best, cost):
    # Deadline check
    if time.time() - _START_TIME > _LIMIT:
        return np.zeros_like(mu)

    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    # Thompson Sampling: Sample y ~ N(mu, sigma^2)
    # We use a standard normal random variable
    z = np.random.normal(0.0, 1.0, size=mu.shape)
    y = mu + sigma * z

    # Cost-aware score: -y / cost
    # Higher score is better.
    # Since y is the predicted function value (lower is better), -y is higher when y is lower.
    # Dividing by cost penalizes expensive evaluations.
    score = -y / cost

    # Ensure finite output
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Replace -inf with a very low finite number to avoid issues if the evaluator doesn't handle inf
    score = np.where(np.isfinite(score), score, -1e9)

    return score
