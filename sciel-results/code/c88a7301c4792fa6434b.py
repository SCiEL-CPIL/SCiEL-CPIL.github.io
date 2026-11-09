"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Structural idea: Knowledge Gradient (KG) approximation that decouples the
    probability of discovering a new best value from the magnitude of that value,
    combined with the refinement machinery of the current LCB-based program
    (probability-weighted exploitation, exploration bonus, soft sqrt-cost penalty,
    and an explicit deadline guard).

    For minimization:
      - EI captures the expected reduction in f_best if x is evaluated.
      - A probability-of-discovery term (P(f(x) < f_best)) weights the exploitation
        term so that regions with high confidence of improvement are favoured even
        when the absolute expected improvement is small (flat Griewank landscape).
      - An exploration bonus rewards high-uncertainty regions where new minima may
        be hidden, scaled by the probability of a significant improvement.
      - The soft sqrt(cost) penalty makes the cost awareness less aggressive than
        linear division, which is important because cost is in [1, 2].
    """
    # Deadline guard: the evaluator kills after 330 s; each call should finish
    # well within 300 s. If we are running long, return a cheap fallback.
    if not hasattr(acquisition, '_start_time'):
        acquisition._start_time = time.time()

    elapsed = time.time() - acquisition._start_time
    if elapsed > 290:
        # Cheap fallback: expected improvement per unit cost
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0) / cost

    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)

    # Standard Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Probability of improvement: P(f(x) < f_best)
    prob_improve = norm.cdf(z)

    # Probability of a significant improvement (delta = 0.01, characteristic
    # scale for Griewank 2D where values are typically in [0, 0.1])
    delta = 0.01
    z_delta = (f_best - delta - mu) / sigma_safe
    prob_significant = norm.cdf(z_delta)

    # Soft cost penalty: sqrt(cost) is in [1, sqrt(2)] ~ [1, 1.414]
    cost_penalty = np.sqrt(cost)

    # KG-style score:
    #   Exploitation: EI weighted by the probability of discovery.
    #   Exploration:  probability of significant improvement times the
    #                 uncertainty (sigma), which rewards high-variance regions
    #                 where a new global minimum may be hidden.
    # Both terms are divided by the soft cost penalty.
    exploitation = ei * prob_improve
    exploration = 0.5 * prob_significant * sigma_safe

    score = (exploitation + exploration) / cost_penalty

    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
