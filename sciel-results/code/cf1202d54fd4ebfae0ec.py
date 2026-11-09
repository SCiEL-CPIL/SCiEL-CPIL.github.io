import time
import numpy as np
from scipy.stats import norm

# Pre-compute lookup tables for speed
_Z_LUT_MIN = -12.0
_Z_LUT_MAX = 12.0
_Z_LUT_N = 100001
_z_lut_grid = np.linspace(_Z_LUT_MIN, _Z_LUT_MAX, _Z_LUT_N)
_cdf_lut = norm.cdf(_z_lut_grid)
_pdf_lut = norm.pdf(_z_lut_grid)


def _fast_norm_cdf(z):
    """Vectorized norm.cdf via interpolation."""
    z = np.clip(z, _Z_LUT_MIN, _Z_LUT_MAX)
    idx = ((z - _Z_LUT_MIN) / (_Z_LUT_MAX - _Z_LUT_MIN) * (_Z_LUT_N - 1)).astype(np.int64)
    idx = np.clip(idx, 0, _Z_LUT_N - 2)
    x0 = _z_lut_grid[idx]
    x1 = _z_lut_grid[idx + 1]
    y0 = _cdf_lut[idx]
    y1 = _cdf_lut[idx + 1]
    t = (z - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def _fast_norm_pdf(z):
    """Vectorized norm.pdf via interpolation."""
    z = np.clip(z, _Z_LUT_MIN, _Z_LUT_MAX)
    idx = ((z - _Z_LUT_MIN) / (_Z_LUT_MAX - _Z_LUT_MIN) * (_Z_LUT_N - 1)).astype(np.int64)
    idx = np.clip(idx, 0, _Z_LUT_N - 2)
    x0 = _z_lut_grid[idx]
    x1 = _z_lut_grid[idx + 1]
    y0 = _pdf_lut[idx]
    y1 = _pdf_lut[idx + 1]
    t = (z - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Combines Expected Improvement (EI) with cost-awareness and adaptive
    exploration. Returns 2000 finite scores (higher = evaluate next).
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-12)

    # Expected Improvement
    z = (f_best - mu) / sigma
    cdf_z = _fast_norm_cdf(z)
    pdf_z = _fast_norm_pdf(z)
    ei = (f_best - mu) * cdf_z + sigma * pdf_z

    # Cost-aware EI: divide by cost to favor cheaper evaluations
    # Cost is in [1, 2], so this normalizes the improvement per unit cost
    ei_cost = ei / (cost + 1e-9)

    # UCB with adaptive beta
    # When f_best is close to the minimum (0), exploit more
    # When f_best is far, explore more
    convergence = max(0.0, min(1.0, 1.0 - f_best / 3.0))
    beta = 2.5 * (1.0 - 0.6 * convergence)  # beta in [1.0, 2.5]

    ucb = mu + beta * sigma
    # Lower UCB is better (we're minimizing)
    ucb_score = -ucb

    # Normalize scores to [0, 1] range for combination
    ei_cost_min = np.min(ei_cost)
    ei_cost_max = np.max(ei_cost)
    ei_cost_range = ei_cost_max - ei_cost_min + 1e-9
    ei_cost_norm = (ei_cost - ei_cost_min) / ei_cost_range

    ucb_score_min = np.min(ucb_score)
    ucb_score_max = np.max(ucb_score)
    ucb_score_range = ucb_score_max - ucb_score_min + 1e-9
    ucb_score_norm = (ucb_score - ucb_score_min) / ucb_score_range

    # Adaptive weighting: more EI when converging, more UCB when exploring
    w_ei = 0.5 + 0.4 * convergence  # [0.5, 0.9]
    w_ucb = 1.0 - w_ei  # [0.1, 0.5]

    # Combined score
    score = w_ei * ei_cost_norm + w_ucb * ucb_score_norm

    # Add small exploration bonus for high uncertainty
    sigma_max = np.max(sigma) + 1e-9
    sigma_norm = sigma / sigma_max
    exploration_bonus = 0.05 * sigma_norm * (1.0 - convergence)
    score += exploration_bonus

    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)

    # Time guard
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        pass

    return score
