"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Batch-aware Thompson Sampling heuristic with cost-awareness.
    # Sample a single function value from the GP posterior for each candidate.
    # Use a deterministic pseudo-random sample based on index for reproducibility
    # and to avoid stochastic variability across calls.
    # The sample is: y = mu - sigma * z, where z ~ N(0,1)
    # We want to maximize expected improvement per unit cost.
    # Improvement: max(0, f_best - y)
    # For a sample y, the improvement is max(0, f_best - y)
    # We use a single sample per candidate (Thompson Sampling style)
    # and divide by cost to make it cost-aware.
    
    # Generate pseudo-random normal samples for Thompson Sampling.
    # Use a seed derived from f_best to ensure the noise pattern evolves
    # as the optimization progresses, helping to escape local optima
    # caused by a static noise realization.
    n = len(mu)
    
    # Derive a seed from f_best. 
    # f_best is a float. We convert it to an integer seed.
    # We use a large multiplier to get unique seeds for small changes in f_best.
    seed = int(abs(f_best) * 1000000) % (2**32)
    rng = np.random.RandomState(seed)
    z = rng.standard_normal(n)
    
    # Sampled function values: lower is better, so we sample y = mu - sigma * z
    # This means when z > 0, we sample a lower (better) value
    y = mu - sigma * z
    
    # Improvement relative to current best
    improvement = np.maximum(0, f_best - y)
    
    # Cost-aware: divide by cost
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    score = improvement / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
