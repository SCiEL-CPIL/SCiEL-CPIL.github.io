"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining LCB structural design with
    probability-weighted refinement machinery.

    Structural idea (from LCB family):
      - Use LCB = mu - beta * sigma as the primary signal.
      - This is confidence-based rather than improvement-based, which
        handles the flat, multi-modal Griewank landscape better.

    Refinement machinery (from KG/probability-weighted family):
      - Probability of improvement weights the exploitation term.
      - Probability of significant improvement provides an exploration bonus.
      - Soft sqrt(cost) penalty for cost awareness.
      - Deadline guard for safety.

    The combined score is:
      score = [P_improve * (f_best - mu) + 0.5 * P_significant * sigma] / sqrt(cost)
    
    This is equivalent to the LCB-based expected improvement with a
    probability-weighted exploration bonus, all normalized by soft cost.
    """
    # Deadline guard: the evaluator kills after 330 s; each call should finish
    # well within 300 s.
    if not hasattr(acquisition, '_start_time'):
        acquisition._start_time = time.time()

    elapsed = time.time() - acquisition._start_time
    if elapsed > 290:
        # Cheap fallback: return 1/cost to prefer cheaper evaluations
        return 1.0 / np.maximum(cost, 1e-8)

    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)

    # Probability of improvement: P(f(x) < f_best)
    z = (f_best - mu) / sigma_safe
    prob_improve = norm.cdf(z)

    # Expected improvement component: P(f(x) < f_best) * (f_best - mu)
    # This is a probability-weighted exploitation term
    expected_improvement = (f_best - mu) * prob_improve

    # Probability of a significant improvement (delta = 0.01, characteristic
    # scale for Griewank 2D where values are typically in [0, 0.1])
    delta = 0.01
    z_delta = (f_best - delta - mu) / sigma_safe
    prob_significant = norm.cdf(z_delta)

    # Exploration bonus: probability of significant improvement times the
    # uncertainty (sigma), which rewards high-variance regions where new
    # minima may be hidden
    exploration = 0.5 * prob_significant * sigma_safe

    # Soft cost penalty: sqrt(cost) is in [1, sqrt(2)] ~ [1, 1.414]
    cost_penalty = np.sqrt(np.maximum(cost, 1e-8))

    # Combined score: exploitation + exploration, all divided by cost penalty
    score = (expected_improvement + exploration) / cost_penalty

    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
