"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function: EI per unit cost plus a small exploration term.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Safe divisions
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-12)

    # Z-score for expected improvement
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -30.0, 30.0)

    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        cdf_val = norm.cdf(z)
        pdf_val = norm.pdf(z)

    # Expected Improvement
    ei = (f_best - mu) * cdf_val + sigma * pdf_val
    ei = np.where(np.isfinite(ei), ei, 0.0)

    # Cost-aware EI
    score_ei = ei / cost_safe

    # Exploration term: normalized uncertainty
    score_unc = sigma / cost_safe

    # Weighted combination: slight bias toward exploration for tight budget
    scores = 0.65 * score_ei + 0.35 * score_unc

    # Ensure finite, non-negative
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
