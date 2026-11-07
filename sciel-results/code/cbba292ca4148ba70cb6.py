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


def _knowledge_gradient(mu, sigma, f_best):
    """
    Knowledge Gradient: probability that evaluating x will lead to a better
    final decision. For 1-step, this is P(f(x) < f_best).
    """
    s = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / s
    kg = norm.cdf(z)
    return np.where(np.isfinite(kg), np.clip(kg, 0.0, 1.0), 0.0)


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    t0 = time.time()
    
    s = np.maximum(sigma, 1e-9)
    c = np.maximum(cost, 1e-9)
    
    # Compute EI and KG
    ei = _expected_improvement(mu, sigma, f_best)
    kg = _knowledge_gradient(mu, sigma, f_best)
    
    # Blend EI and KG with cost-awareness
    # Use a softer cost penalty (power 0.5) to allow exploration of moderately expensive points
    # Weight EI more heavily than KG since we care about actual improvement
    # The blend: 0.7 * EI + 0.3 * KG, all divided by cost^0.5
    blended = 0.7 * ei + 0.3 * kg * (f_best + 1e-9)  # Scale KG by f_best to match EI scale
    
    score = blended / (c ** 0.5)
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Safety margin: if we've spent too much time, return immediately
    elapsed = time.time() - t0
    if elapsed > 280:  # 20s buffer before 300s limit
        return score
    
    return score
# EVOLVE-BLOCK-END
