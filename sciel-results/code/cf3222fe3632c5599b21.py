"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard: the evaluator kills after 330s.
# We track elapsed time and ensure each call returns promptly.
_DEADLINE_START = time.time()
_DEADLINE_LIMIT = 320.0  # safety margin: 10s before the 330s kill


def _remaining_time():
    return max(0.0, _DEADLINE_LIMIT - (time.time() - _DEADLINE_START))


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition with diversity-aware penalty.
    
    Parameters:
        mu: (2000,) GP mean in objective units
        sigma: (2000,) GP std in objective units
        f_best: scalar, lowest observed value
        cost: (2000,) evaluation cost for each candidate
    
    Returns:
        (2000,) scores, higher = evaluate next
    """
    # Deadline guard: if we're running out of time, return a safe default
    if _remaining_time() < 2.0:
        # Fallback: just return EI/cost without any extra computation
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        scores = ei / np.maximum(cost, 1e-9)
        return np.where(np.isfinite(scores), scores, 0.0)

    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # --- Core: Expected Improvement per unit cost ---
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # Ensure non-negative EI (numerical guard)
    ei = np.maximum(ei, 0.0)
    ei_cost = ei / cost_safe

    # --- Diversity penalty: penalize candidates in low-sigma (well-sampled) regions ---
    # Low sigma means the GP is confident (many nearby evaluations), so exploring there
    # is redundant. We use a smooth penalty based on sigma:
    #   penalty = alpha * (1 / (sigma + eps))  -- but this can blow up.
    # Instead, use a bounded soft penalty:
    #   penalty = alpha * exp(-sigma / sigma_scale)
    # This is high when sigma is small (well-sampled) and approaches 0 when sigma is large.
    sigma_scale = 0.05  # tune: how quickly the penalty decays with sigma
    diversity_penalty = np.exp(-sigma_safe / sigma_scale)
    
    # Normalize diversity penalty to [0, 1] range for stable scaling
    # The max possible penalty is 1.0 (when sigma -> 0)
    
    # Blend: subtract a fraction of the diversity penalty from EI/cost
    # We want the penalty to be meaningful relative to the EI scale.
    # Scale the penalty by a factor of the median EI/cost to keep it in the same units.
    ei_median = np.median(ei_cost)
    if ei_median > 1e-12:
        penalty_weight = 0.3  # fraction of median EI/cost to use as penalty scale
        penalty = penalty_weight * ei_median * diversity_penalty
    else:
        penalty = np.zeros_like(ei_cost)

    scores = ei_cost - penalty

    # --- Additional exploration boost for high-uncertainty regions ---
    # UCB-like term: adds exploration for high sigma
    beta = 0.5  # exploration weight
    ucb_term = beta * sigma_safe / cost_safe
    scores = scores + ucb_term

    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
