"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Improved Batch-aware Thompson Sampling with cost-awareness.
    # 1. Derive a dynamic seed from f_best to change the noise pattern
    #    as the optimization progresses, preventing stagnation in local optima
    #    defined by a fixed noise realization.
    # 2. Sample y ~ N(mu, sigma) for each candidate.
    # 3. Compute cost-aware improvement: max(0, f_best - y) / cost.
    
    n = len(mu)
    
    # Derive seed from f_best to ensure the "random" samples change
    # as the best observed value improves. This mimics the stochastic nature
    # of Thompson Sampling across iterations while keeping the function
    # deterministic for a given state (mu, sigma, f_best, cost).
    seed_val = int(np.abs(f_best) * 1e9) % (2**31 - 1)
    rng = np.random.RandomState(seed_val)
    z = rng.standard_normal(n)
    
    # Sample function values. y = mu + sigma * z
    # Since we are minimizing, lower y is better.
    y = mu + sigma * z
    
    # Calculate improvement over current best
    # improvement is positive only if y < f_best
    improvement = np.maximum(0.0, f_best - y)
    
    # Cost-aware score
    # Cost is given as 1 + mean(x_scaled), so it is >= 1.
    # Dividing by cost penalizes expensive candidates.
    cost_safe = np.maximum(cost, 1e-9)
    score = improvement / cost_safe
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
