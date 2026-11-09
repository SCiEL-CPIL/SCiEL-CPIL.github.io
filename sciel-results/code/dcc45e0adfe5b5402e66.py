"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_LIMIT = 285.0  # Safety margin below 300s


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a Knowledge Gradient-inspired approach with exploitation bias.
    """
    # Deadline check
    if time.time() - _START_TIME > _LIMIT:
        return np.zeros_like(mu)

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    n = mu.shape[0]

    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    # Avoid division by zero in sigma
    sigma_safe = np.maximum(sigma, 1e-8)

    # Compute z-score for improvement probability
    z = (f_best - mu) / sigma_safe
    # Clip z to prevent numerical underflow/overflow in pdf/cdf
    z = np.clip(z, -10, 10)

    cdf = norm.cdf(z)
    pdf = norm.pdf(z)

    # Expected Improvement (EI)
    ei = (f_best - mu) * cdf + sigma_safe * pdf

    # Knowledge Gradient core:
    # The expected reduction in f_best from evaluating x is EI.
    # But KG emphasizes the *probability* of finding a new best times the
    # *conditional expected improvement* given that it IS a new best.
    #
    # P(new best) = cdf
    # E[improvement | new best] = ei / cdf  (when cdf > 0)
    # KG ≈ cdf * (ei / cdf) = ei  ... so pure EI IS the 1-point KG.
    #
    # However, in practice, a more robust variant weights the exploitation
    # component (low mu) by the discovery probability to avoid chasing
    # high-mu, high-sigma points that have low probability of actually
    # being the global minimum.
    #
    # The key structural insight: gate the exploitation bias by discovery
    # probability. Points that are likely to be new bests get stronger
    # exploitation pull toward lower mu. Points unlikely to improve get
    # less exploitation pressure, allowing exploration via the sigma*pdf term.

    # Gated exploitation: only push toward low mu when discovery is likely
    # Use a soft gate: sigmoid-like via cdf itself
    # Reduced bias to 0.1 to balance exploration and exploitation,
    # preventing premature convergence to local minima in the multi-modal
    # Griewank landscape.
    exploitation = -0.1 * mu * cdf

    # Cost damping: reduce sensitivity to cost so search can reach
    # the central region (inner 80%) where the minimum is constrained
    # cost ranges from 1.0 (edges) to ~2.0 (center)
    # Dampening: 1.0 + 0.5*(cost-1.0) maps [1,2] -> [1, 1.5]
    cost_damped = 1.0 + 0.5 * (cost - 1.0)

    # Combine: KG-style EI + gated exploitation, divided by damped cost
    score = (ei + exploitation) / cost_damped

    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
