"""
Strategy: Cost-Aware Thompson Sampling (TS) with Exponential Cost Penalty.

This strategy diverges from the current EI/LCB blend by using Thompson Sampling,
a stochastic method that samples from the posterior distribution of the objective
function. This provides a natural balance between exploration and exploitation
without requiring a fixed exploration parameter (beta). The stochasticity helps
escape local optima in the multimodal Powell 4D landscape.

To account for the non-uniform evaluation costs, the sampled values are penalized
exponentially by the cost. This strongly discourages evaluating expensive points
unless the posterior mean is very promising, effectively steering the search
towards cheaper regions of the domain while still allowing for occasional
exploration of expensive areas if the uncertainty is high.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Handle edge cases
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    cost_safe = np.where(cost > 1e-9, cost, 1e-9)
    
    # Thompson Sampling: Sample from the posterior N(mu, sigma)
    # We want to minimize f, so we look for low sampled values.
    y_samples = np.random.normal(mu, sigma_safe)
    
    # Cost-aware scoring:
    # We want high scores for low y and low cost.
    # Score = -y * exp(-cost)
    # The exponential penalty ensures that even if y is very low,
    # a high cost will significantly reduce the score.
    # This is more aggressive than linear division in penalizing high costs.
    cost_penalty = np.exp(-cost_safe)
    scores = -y_samples * cost_penalty
    
    # Ensure finite values
    if not np.all(np.isfinite(scores)):
        # Fallback to EI / cost if numerical issues arise
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        scores = ei / cost_safe
    
    # Time guard
    elapsed = time.time() - start_time
    if elapsed > 5.0:
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return ei / cost_safe

    return scores
# EVOLVE-BLOCK-END
