"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_EPS = 1e-12

def _ackley_2d(x):
    """Ackley 2D function (standard, minimum at 0,0)."""
    x1 = x[:, 0]
    x2 = x[:, 1]
    sum_sq = x1**2 + x2**2
    sum_cos = np.cos(2.0 * np.pi * x1) + np.cos(2.0 * np.pi * x2)
    return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * sum_sq)) - np.exp(0.5 * sum_cos) + np.e + 20.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling for Bayesian optimization of Ackley 2D.
    
    Samples from the GP posterior at each candidate, ranks by sampled value / cost,
    and returns rank-based scores. This implicitly balances exploration and cost-efficiency.
    
    mu, sigma: (2000,) arrays of GP mean and std in objective units
    f_best: scalar, lowest observed value
    cost: (2000,) array of evaluation costs
    
    Returns: (2000,) finite scores, higher = prefer to evaluate next
    """
    start_time = time.time()
    deadline = start_time + 250.0  # Safety margin before 300s limit (killed at 330s)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, _EPS)
    cost_safe = np.maximum(cost, _EPS)
    
    # Thompson Sampling: sample from the posterior at each candidate
    # Use a seeded RNG based on f_best to ensure different samples each call
    # but deterministic within a single call
    seed_val = int(abs(hash(f_best)) % (2**31))
    rng = np.random.default_rng(seed_val)
    
    # Sample objective values from N(mu, sigma^2) for each candidate
    sampled_values = mu + sigma * rng.standard_normal(n)
    
    # Score: lower sampled value / cost is better
    # We want higher score = better, so use negative of (sampled_value / cost)
    ts_score = -sampled_values / cost_safe
    
    # Rank-based score: rank candidates by ts_score (higher ts_score = better)
    # Use argsort to get ranks, then invert so that best candidate gets highest score
    order = np.argsort(ts_score)  # ascending: worst to best
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n)  # worst gets 0, best gets n-1
    
    # Normalize ranks to [0, 1] for stable scoring
    if n > 1:
        score = ranks / (n - 1)
    else:
        score = np.ones(n, dtype=np.float64)
    
    # Check deadline
    if time.time() > deadline:
        pass
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
