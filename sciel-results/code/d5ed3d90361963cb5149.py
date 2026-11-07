"""Acquisition function for Bayesian optimization of Hartmann 3D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard (330s wall-clock limit; use 310s as safety margin)
_DEADLINE = time.time() + 310.0


def _time_left():
    return _DEADLINE - time.time()


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: a hybrid of Expected Improvement (EI) and
    Upper Confidence Bound (UCB) with a diversity-promoting penalty
    in the (mu, sigma) feature space to encourage exploration of
    under-sampled regions of the posterior.

    mu: (2000,) array of GP mean predictions
    sigma: (2000,) array of GP std-dev predictions
    f_best: float, best observed value (lower is better)

    Returns: (2000,) array of finite scores; higher = evaluate next.
    """
    # Safety: if we're close to the deadline, return a simple EI score
    if _time_left() < 2.0:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    sigma_safe = np.maximum(sigma, 1e-9)

    # --- Component 1: Expected Improvement ---
    z_ei = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma * norm.pdf(z_ei)

    # --- Component 2: UCB (beta=1.96 ≈ 95% confidence) ---
    ucb = mu - 1.96 * sigma_safe  # negative because we minimize

    # --- Component 3: Diversity penalty in (mu, sigma) feature space ---
    # Penalize candidates that cluster in the same (mu, sigma) region.
    # Use a simple kernel-density-like penalty: for each candidate,
    # compute the sum of Gaussian kernels over all other candidates
    # in the 2D (mu, sigma) space, and subtract a scaled penalty.
    # This encourages the acquisition to spread across different
    # posterior characteristics.

    # Normalize mu and sigma to [0,1] range for the kernel
    mu_min, mu_max = mu.min(), mu.max()
    sigma_min, sigma_max = sigma_safe.min(), sigma_safe.max()
    mu_range = max(mu_max - mu_min, 1e-9)
    sigma_range = max(sigma_max - sigma_min, 1e-9)

    mu_norm = (mu - mu_min) / mu_range
    sigma_norm = (sigma_safe - sigma_min) / sigma_range

    # To keep computation tractable for 2000 points, use a subsampled
    # reference set for the density estimation (e.g., 200 points)
    n_ref = 200
    if len(mu) > n_ref:
        ref_idx = np.linspace(0, len(mu) - 1, n_ref, dtype=int)
        mu_ref = mu_norm[ref_idx]
        sigma_ref = sigma_norm[ref_idx]
    else:
        mu_ref = mu_norm
        sigma_ref = sigma_norm

    # Compute pairwise distances in (mu_norm, sigma_norm) space
    # For each of the 2000 candidates, compute distance to each reference point
    # Use squared Euclidean distance
    # (2000, n_ref) matrix
    dmu = mu_norm[:, None] - mu_ref[None, :]
    dsigma = sigma_norm[:, None] - sigma_ref[None, :]
    dist_sq = dmu ** 2 + dsigma ** 2

    # Gaussian kernel with bandwidth ~0.15
    bandwidth = 0.15
    density = np.exp(-dist_sq / (2 * bandwidth ** 2))
    density_sum = density.sum(axis=1)  # (2000,)

    # Normalize density to [0, 1]
    d_max = density_sum.max()
    if d_max > 0:
        density_norm = density_sum / d_max
    else:
        density_norm = np.zeros_like(density_sum)

    # Diversity penalty: subtract a small multiple of the normalized density
    diversity_weight = 0.15
    diversity_term = -diversity_weight * density_norm

    # --- Combine components ---
    # Normalize EI and UCB to comparable scales
    ei_min, ei_max = ei.min(), ei.max()
    ei_range = max(ei_max - ei_min, 1e-9)
    ei_norm = (ei - ei_min) / ei_range

    ucb_min, ucb_max = ucb.min(), ucb.max()
    ucb_range = max(ucb_max - ucb_min, 1e-9)
    ucb_norm = (ucb - ucb_min) / ucb_range

    # Weighted combination
    w_ei = 0.5
    w_ucb = 0.3
    w_div = 0.2

    score = w_ei * ei_norm + w_ucb * ucb_norm + w_div * (-density_norm)

    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    return score
# EVOLVE-BLOCK-END
