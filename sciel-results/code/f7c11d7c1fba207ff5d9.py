"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Hard budget-aware gate:
    #   - Compute Expected Improvement (EI) for each candidate.
    #   - Allow a candidate only if its cost is <= threshold.
    #   - threshold adapts to f_best: early on it is tight (favor low cost),
    #     later it relaxes as the objective improves.
    #
    # This replaces the smooth EI/cost tradeoff with a feasibility constraint,
    # forcing exploitation of the low-cost corner until GP uncertainty there is
    # exhausted, then gradually widening the acceptable cost range.
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Base cost of any evaluation is 1.0 (from 1 + mean(x_scaled)).
    # The variable part is mean(x_scaled) in [0, 1].
    # So cost is in [1, 2].
    
    # Adaptive threshold:
    #   - Start very tight: only the cheapest ~10% of the cost range.
    #   - As f_best improves (gets closer to 0), relax the threshold.
    #   - f_best is positive (Levy 2D minimum is 0), so use f_best directly.
    #   - Threshold in cost units: 1 + alpha * f_best, where alpha controls relaxation.
    #     alpha = 0.3 means when f_best = 1.0, threshold = 1.3 (allow moderate cost).
    #     When f_best = 0.1, threshold = 1.03 (still very tight).
    #     When f_best = 0.01, threshold = 1.003 (almost only cheapest).
    #
    # But we also want to ensure we don't get stuck: if f_best is large (> 5),
    # cap the threshold at 1.5 to avoid being too restrictive early on.
    alpha = 0.3
    threshold = 1.0 + alpha * f_best
    threshold = np.minimum(threshold, 1.5)
    
    # Hard gate: only score candidates with cost <= threshold
    # Use a large negative penalty for infeasible candidates
    feasible = cost <= threshold
    penalty = -1e6
    
    scores = np.where(feasible, ei, penalty)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e6)
    
    return scores
# EVOLVE-BLOCK-END
