"""Thompson Sampling acquisition for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling-like acquisition:
    1. Sample a single latent objective value for each candidate from the GP posterior: y = mu + sigma * z, where z ~ N(0, 1).
    2. Compute a cost-aware score by dividing the negative sampled value by the evaluation cost.
       (We minimize the objective, so lower sampled values are better; we invert the sign to maximize score.)
    3. Return the 2000 scores; the evaluator picks the candidate with the highest score.
    
    This differs from EI/PI by sampling from the full predictive distribution,
    which naturally balances exploration (high sigma regions can produce low samples)
    and exploitation (low mu regions tend to produce low samples).
    """
    t0 = time.time()
    deadline = t0 + 285.0  # safety margin before 300s limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Guard against zero or negative sigma and cost
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Sample one latent value per candidate from the GP posterior
    z = np.random.standard_normal(mu.shape)
    sampled_values = mu + sigma_safe * z

    # Thompson sampling for minimization: prefer low sampled values.
    # Score = -sampled_value / cost  (higher score = better candidate to evaluate next)
    score = -sampled_values / cost_safe

    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)

    # Time guard: if we're close to deadline, return immediately
    if time.time() > deadline - 5.0:
        return score

    return score
# EVOLVE-BLOCK-END
