"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Time budget guard constants
_TIME_LIMIT = 300.0
_START_TIME = time.time()

def _check_time():
    return time.time() - _START_TIME < _TIME_LIMIT

def acquisition(mu, sigma, f_best, cost):
    # Ensure we have a starting time if this is the first call
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        
    # If we are running out of time, return a safe fallback (exploit current best)
    if not _check_time():
        # Return something that prioritizes low mu, but safe. 
        # Actually, if we are close to limit, we should just return the seed logic or a simple exploit.
        # The evaluator kills after 330s. We have 300s budget.
        # If we hit the guard, we must return *something* valid.
        # Let's return a simple EI to be safe and fast.
        z = (f_best - mu) / (sigma + 1e-9)
        return ((f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)) / cost

    n_samples = 500
    n_cands = len(mu)
    
    # Vectorized Monte Carlo Knowledge Gradient
    # We simulate the objective values of all candidates and the current best.
    # y_cand ~ N(mu, sigma^2)
    # y_best ~ N(f_best, 0) -- actually f_best is observed, but in KG we often treat the current best 
    # as a random variable if we haven't seen the true minimum, or just compare to the random draws.
    # Standard KG: P(candidate i is the global min) = E[ 1(y_i < min(y_j for j != i)) ]
    # Since we have 2000 candidates, checking all pairs is O(N^2 * S) which is 2000^2 * 500 = 2e9. Too slow.
    
    # Approximation: 
    # 1. We can sample S times.
    # 2. For each sample, we draw y_i for all i.
    # 3. We find the global minimum index of the sample.
    # 4. The KG value for candidate i is the fraction of samples where i was the global min.
    # This is O(N * S). 2000 * 500 = 1e6 operations. Very fast.
    
    # However, we must account for cost.
    # The objective is to minimize Simple Regret.
    # Cost-aware KG: Maximize P(i is best) / cost_i ? Or incorporate cost into the "value" of the candidate?
    # The prompt says: "Cost-aware acquisition function".
    # Current seed: EI / cost.
    # Let's try: KG_score = P(i is global min) / cost_i.
    
    # Generate samples: (n_samples, n_cands)
    # y = mu + sigma * Z
    Z = np.random.randn(n_samples, n_cands)
    y = mu[None, :] + sigma[None, :] * Z
    
    # Find the index of the minimum value in each sample row
    min_indices = np.argmin(y, axis=1)
    
    # Count how often each candidate was the minimum
    counts = np.zeros(n_cands)
    np.add.at(counts, min_indices, 1)
    
    # Probability that candidate i is the global minimum
    p_best = counts / n_samples
    
    # Cost-aware scoring
    # We want to balance finding the global min vs cost.
    # Dividing by cost is a direct heuristic.
    scores = p_best / cost
    
    # Add a small exploration bonus for high uncertainty? 
    # KG naturally explores because if sigma is high, the probability mass is spread, 
    # but if mu is low, it concentrates. 
    # However, pure KG can be too greedy. 
    # Let's blend with a small amount of UCB or EI to ensure we don't get stuck in local minima 
    # or ignore high-variance regions that might contain the true minimum.
    
    # Let's compute a quick EI as a fallback/blender
    z = (f_best - mu) / (sigma + 1e-9)
    ei = ((f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)) / cost
    
    # Blend: 80% KG, 20% EI
    # This ensures we have a robust signal.
    scores = 0.8 * scores + 0.2 * ei
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
