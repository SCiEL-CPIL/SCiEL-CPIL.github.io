"""Cost-aware acquisition for Bayesian optimization (styblinski tang 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: keep a safety margin before the 300s per-evaluation limit.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # 10s safety margin


def _elapsed_ok():
    return time.time() < _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Uncertainty-Weighted Improvement (UWI)
    with a local baseline estimated from GP predictions.

    Instead of using the global f_best, we estimate a local baseline for each
    candidate by taking the minimum of mu over nearby candidates (using a
    kernel-weighted approach). This targets the boundaries of explored basins
    rather than just the global best or pure uncertainty.

    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best (lowest) objective value observed so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate, in the same units as the budget.

    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values are preferred.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-12)

    # If we are close to the deadline, use a simple, fast heuristic.
    if not _elapsed_ok():
        scores = -mu / cost_safe
        scores += 1e-6 * sigma_safe / cost_safe
        return np.where(np.isfinite(scores), scores, -1e30)

    # Estimate a local baseline: for each candidate, the local baseline is
    # a soft minimum of mu values, weighted by how close the GP predictions
    # are in function space. We use a simple approach: the local baseline
    # is the minimum of mu over a neighborhood defined by similar uncertainty
    # levels. Since we don't have coordinates, we use the GP posterior
    # structure: candidates with similar sigma are likely in similar regions.

    # Strategy: For each candidate i, the local baseline is the minimum mu
    # among candidates j where sigma_j is within a factor of 2 of sigma_i.
    # This is O(n^2) but n=2000, so it's feasible.

    # More efficient: use histogram-based binning on sigma to approximate
    # local neighborhoods.

    # Binning approach:
    # 1. Bin sigma values into quantiles.
    # 2. For each bin, compute the minimum mu in that bin and adjacent bins.
    # 3. Use that as the local baseline.

    n = len(mu)
    n_bins = 20
    sigma_min = np.min(sigma_safe)
    sigma_max = np.max(sigma_safe)

    # Create bins based on sigma quantiles for better resolution
    if sigma_max - sigma_min < 1e-12:
        # All sigmas are similar; use global best as local baseline
        local_baseline = np.full(n, f_best)
    else:
        # Quantile-based binning
        bin_edges = np.quantile(sigma_safe, np.linspace(0, 1, n_bins + 1))
        bin_indices = np.digitize(sigma_safe, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        # For each bin, compute the minimum mu
        bin_min_mu = np.full(n_bins, np.inf)
        for b in range(n_bins):
            mask = bin_indices == b
            if np.any(mask):
                bin_min_mu[b] = np.min(mu[mask])

        # Local baseline: minimum of mu in current bin and adjacent bins
        local_baseline = np.zeros(n)
        for b in range(n_bins):
            # Include current bin and adjacent bins
            neighbors = [b]
            if b > 0:
                neighbors.append(b - 1)
            if b < n_bins - 1:
                neighbors.append(b + 1)
            local_min = np.min(bin_min_mu[neighbors])
            mask = bin_indices == b
            local_baseline[mask] = local_min

        # Blend with global best: use the lower of local baseline and f_best
        # This ensures we never ignore the global best
        local_baseline = np.minimum(local_baseline, f_best)

    # Uncertainty-Weighted Improvement (UWI):
    # UWI = (baseline - mu) * Phi(z) + sigma * phi(z) * w
    # where z = (baseline - mu) / sigma
    # and w is a weight that emphasizes uncertainty relative to the local scale.

    z = (local_baseline - mu) / sigma_safe
    z = np.clip(z, -50, 50)

    # Standard EI-like term with local baseline
    ei_local = (local_baseline - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Uncertainty weight: reward candidates where sigma is high relative to
    # the local variation in mu. Compute local scale as std of mu in the
    # neighborhood.
    # Use a simpler proxy: normalize sigma by the global scale of mu.
    mu_range = np.max(mu) - np.min(mu)
    if mu_range < 1e-12:
        mu_range = 1.0

    # UWI: combine EI with an uncertainty bonus that scales with sigma
    # relative to the problem scale
    uncertainty_bonus = sigma_safe / (mu_range + sigma_safe)

    # Combine: primary signal is EI with local baseline, secondary is uncertainty
    # The uncertainty bonus encourages exploring regions with high posterior
    # variance, especially at basin boundaries.
    scores = ei_local + 0.5 * uncertainty_bonus * sigma_safe

    # Cost-aware: divide by cost
    scores = scores / cost_safe

    # Ensure finite output.
    scores = np.where(np.isfinite(scores), scores, -1e30)

    return scores
# EVOLVE-BLOCK-END
