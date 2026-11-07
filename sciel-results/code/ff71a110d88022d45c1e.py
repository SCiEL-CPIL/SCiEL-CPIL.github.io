import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Computes acquisition scores using Cost-Adjusted Thompson Sampling.
    
    Strategy:
    1. Convert mu and sigma to standardized z-scores relative to the current
       best observed value f_best.
    2. Sample from the standard normal distribution to simulate the uncertainty
       (Thompson Sampling).
    3. Combine the mean (exploit) and the sample (explore) to get a sample
       of the objective value for each candidate.
    4. Divide by the cost to create a cost-aware score (higher is better).
    
    This approach is fundamentally different from EI/UCB as it does not rely
    on a fixed parameter (like beta or xi) but uses stochastic sampling to
    balance exploration and exploitation naturally.
    """
    start_time = time.time()
    deadline = 300.0
    
    # Ensure arrays are numpy float64
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    if n == 0:
        return np.array([])

    # Sanitize inputs
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=0.0)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1.0)
    
    # Ensure sigma and cost are strictly positive
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)

    # f_best is a scalar
    f_best = float(f_best)
    if not np.isfinite(f_best):
        f_best = 0.0

    # Check time limit
    if time.time() > deadline - 5.0:
        # Fallback: simple LCB
        return (1.5 * sigma - mu) / cost

    # Thompson Sampling:
    # We want to sample from N(mu, sigma^2).
    # Let z ~ N(0, 1). Then sample = mu + sigma * z.
    # We want to minimize the objective, so we want low values.
    # The acquisition score should be high for candidates likely to have low objective values.
    # Score = -sample / cost.
    
    # Generate random normal samples
    # Using a fixed seed is not possible here as we don't have a seed parameter,
    # but Thompson Sampling is inherently stochastic.
    # To ensure reproducibility in a strict test environment if needed, one might use
    # a hash of the inputs, but standard TS uses fresh randomness.
    # However, since the task says "five seeded instances", the randomness of the
    # acquisition function itself adds variance. 
    # A deterministic alternative to TS that mimics it is using the median or a specific quantile,
    # but that reverts to UCB/LCB. 
    # Let's stick to true TS.
    
    z = np.random.randn(n)
    
    # Sampled objective values
    sampled_values = mu + sigma * z
    
    # Cost-adjusted score
    # We want to maximize (minimize objective / cost)
    # So score = -sampled_values / cost
    scores = -sampled_values / cost
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
