"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Expected Improvement with an adaptive exploration bonus.
    # mu, sigma: (2000,) arrays of GP mean and std (objective units)
    # f_best: scalar, lowest observed value
    # cost: (2000,) array of evaluation costs

    # Time guard: if we're taking too long, return a simple fallback
    t0 = time.time()

    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)

    # Standard Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Clip EI to non-negative
    ei = np.maximum(ei, 0.0)

    # Cost normalization: divide by cost to get EI per unit cost
    cost_safe = np.maximum(cost, eps)
    score = ei / cost_safe

    # Add a small exploration bonus based on uncertainty,
    # but keep it modest so exploitation still dominates.
    # The bonus encourages visiting high-uncertainty regions
    # that might contain better minima, especially early on.
    # We use a relative bonus: sigma / (sigma + 1) scaled down.
    # Actually, let's use a simpler approach: add a small multiple of sigma/cost
    # to break ties in favor of uncertain points, but weight it by how far
    # we are from convergence.

    # A better structural idea: use a hybrid of EI and a pure exploration term
    # weighted by the ratio of current best to the scale of the problem.
    # Since we don't know the problem scale directly, use f_best as a proxy.
    # If f_best is large relative to sigma, we're still exploring; if small, exploit.

    # Let's compute a simple adaptive weight:
    # When f_best is large (early), weight exploration more.
    # When f_best is small (late), weight exploitation more.
    # Use: weight = 1 / (1 + f_best / (mean_sigma + eps))
    # where mean_sigma is the average sigma across candidates.

    mean_sigma = np.mean(sigma_safe)
    if mean_sigma > eps:
        # Explore more when f_best is large relative to uncertainty
        explore_weight = 1.0 / (1.0 + max(f_best, 0.0) / (mean_sigma + eps))
        # explore_weight is in (0, 1); higher when f_best is small relative to sigma
        # Actually, we want the opposite: explore more when f_best is LARGE (early).
        # Let's flip: explore_weight = f_best / (f_best + mean_sigma + eps)
        explore_weight = max(f_best, 0.0) / (max(f_best, 0.0) + mean_sigma + eps)
        # When f_best >> mean_sigma (early), explore_weight ~ 1 (more exploration)
        # When f_best << mean_sigma (late), explore_weight ~ 0 (more exploitation)
    else:
        explore_weight = 0.5

    # Exploration term: sigma / cost (uncertainty per unit cost)
    explore_score = sigma_safe / cost_safe

    # Combine: (1 - explore_weight) * exploitation + explore_weight * exploration
    # But we need to balance the scales. EI and sigma/cost may have different magnitudes.
    # Let's normalize both to have similar scale.
    ei_max = np.max(ei) + eps
    exp_max = np.max(explore_score) + eps
    ei_norm = ei / ei_max
    exp_norm = explore_score / exp_max

    score = (1.0 - explore_weight) * ei_norm + explore_weight * exp_norm

    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)

    # Time check
    if time.time() - t0 > 280:
        # Fallback: just return EI/cost
        return np.where(np.isfinite(ei / cost_safe), ei / cost_safe, 0.0)

    return score
# EVOLVE-BLOCK-END
