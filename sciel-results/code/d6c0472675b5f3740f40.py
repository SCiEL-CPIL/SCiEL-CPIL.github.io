import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Rastrigin 2D.
    Strategy: Thompson Sampling with Cost-Adjusted Standard Normal Sampling.
    Instead of deterministic EI/UCB, we sample from the posterior and divide by cost.
    This implicitly balances exploration and exploitation based on the posterior distribution.
    """
    start_time = time.time()
    deadline = 300.0

    # Ensure arrays are numpy
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    
    # Basic sanitization
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=0.0)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1.0)
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)

    # f_best is a scalar
    f_best = float(f_best)
    if not np.isfinite(f_best):
        f_best = 0.0

    # Thompson Sampling:
    # Sample from N(mu, sigma^2)
    # We want to minimize the objective, so we want to pick the point with the lowest sampled value.
    # However, the acquisition function returns scores where HIGHER is better.
    # So we return -sample / cost.
    
    # Use a fixed random state for determinism within the call? 
    # Actually, using global numpy random state is fine, but let's use a local RNG for safety?
    # The task says "seeded instances", but the acquisition function itself doesn't get a seed.
    # Standard practice in these loops is that the GP is refit each time, so the randomness
    # in acquisition is part of the stochastic search.
    
    # To ensure reproducibility across calls if the same inputs are provided (unlikely in a loop),
    # we might want to seed based on inputs, but usually just using np.random is fine.
    # However, to be safe against any hidden state issues, let's just use the global random state.
    
    samples = mu + sigma * np.random.standard_normal(n)
    
    # Score is negative sampled value divided by cost
    # Higher score = lower sampled value / lower cost
    scores = -samples / cost
    
    # Check validity
    if not np.all(np.isfinite(scores)):
        # Fallback to EI/Cost if Thompson sampling fails (e.g. extreme values)
        z = (f_best - mu) / sigma
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        scores = ei / cost
        if not np.all(np.isfinite(scores)):
            return np.zeros(n)

    # Time check
    if time.time() - start_time > deadline:
         return scores

    return scores
