"""Rewritten acquisition for Bayesian optimization of sphere 1d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_T0 = time.time()
_DEADLINE = _T0 + 315.0  # safety margin vs 330s kill


def acquisition(mu, sigma, f_best):
    # Deadline guard: if we're getting close to the limit, return a safe fallback
    elapsed = time.time() - _T0
    if elapsed > _DEADLINE:
        # Return scores that favor low mu (exploitation) to be safe
        return -mu.copy()

    # Ensure sigma is positive
    s = np.maximum(sigma, 1e-8)

    # Use a hybrid acquisition: combine Expected Improvement (EI) and
    # Upper Confidence Bound (UCB) to balance exploitation and exploration.
    #
    # EI: standard expected improvement for minimization
    # UCB: mu - kappa * sigma (lower is better, so we negate for "higher is better" score)
    #
    # For sphere function on [-5,5], we want to balance:
    # - Exploitation: points with low mu (close to f_best)
    # - Exploration: points with high sigma (uncertain regions)
    #
    # We'll use a weighted combination:
    # score = alpha * EI + (1 - alpha) * UCB_score
    # where UCB_score = -(mu - kappa * sigma) = -mu + kappa * sigma
    #
    # But actually, for minimization, UCB is: mu - kappa * sigma (we want to minimize this)
    # So the "score" (higher is better) would be: -(mu - kappa * sigma) = -mu + kappa * sigma
    #
    # Let's compute both and combine them.

    # Expected Improvement
    z = (f_best - mu) / s
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + s * pdf_z
    ei = np.where(np.isfinite(ei), ei, 0.0)
    ei = np.maximum(ei, 0.0)

    # UCB score (higher is better for selection)
    # For minimization: UCB value = mu - kappa * sigma
    # Score (higher is better) = -UCB value = -mu + kappa * sigma
    kappa = 1.0  # exploration-exploitation trade-off
    ucb_score = -mu + kappa * s

    # Normalize both to [0, 1] range for fair combination
    # EI is already non-negative
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)

    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-12:
        ucb_norm = (ucb_score - ucb_min) / ucb_range
    else:
        ucb_norm = np.zeros_like(ucb_score)

    # Weighted combination: favor exploration early, exploitation later
    # Since we don't know the step, use a fixed weight that balances both
    alpha = 0.5  # 50% EI, 50% UCB
    score = alpha * ei_norm + (1 - alpha) * ucb_norm

    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
