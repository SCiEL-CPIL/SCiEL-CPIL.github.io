"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _expected_improvement(mu, sigma, f_best):
    """
    Expected Improvement: E[max(f_best - f(x), 0)]
    where f(x) ~ N(mu, sigma^2).
    """
    s = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / s
    ei = (f_best - mu) * norm.cdf(z) + s * norm.pdf(z)
    return np.where(np.isfinite(ei), np.maximum(ei, 0.0), 0.0)


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    t0 = time.time()
    
    s = np.maximum(sigma, 1e-9)
    c = np.maximum(cost, 1e-9)
    
    # Compute Expected Improvement
    ei = _expected_improvement(mu, sigma, f_best)
    
    # Thinning-based approximation: subtract a fraction of the top-k EIs
    # to model the probability that this point will remain the best after
    # a batch of evaluations. This shifts from immediate local improvement
    # to a more global view.
    k = 5
    if ei.size > k:
        # Get the top-k EI values
        top_k_ei = np.partition(ei, -k)[-k:]
        # Subtract a fraction of the mean top-k EI to penalize points
        # that are likely to be surpassed by other high-EI points
        # This is a structural change from simple EI
        thinning_penalty = 0.1 * np.mean(top_k_ei)
        ei_thinned = ei - thinning_penalty
        ei_thinned = np.maximum(ei_thinned, 0.0)
    else:
        ei_thinned = ei
    
    # Apply cost-aware penalty with softened exponent
    # Using cost^0.8 allows selecting points with higher potential improvement
    # even if slightly more expensive, while still respecting the budget
    score = ei_thinned / (c ** 0.8)
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Safety margin: if we've spent too much time, return immediately
    elapsed = time.time() - t0
    if elapsed > 280:  # 20s buffer before 300s limit
        return score
    
    return score
# EVOLVE-BLOCK-END
