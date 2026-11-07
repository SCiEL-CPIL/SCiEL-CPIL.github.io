"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Global deadline guard: keep a safety margin before the 330 s kill.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 300.0  # 30 s margin


def _time_guard():
    """Return True if we are within the safe time budget."""
    return time.time() < _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement with
    adaptive confidence and cost penalty.
    Returns 2000 finite scores; higher = evaluate next.
    """
    # Deadline guard: if running out of time, return a simple valid fallback.
    if not _time_guard():
        c = np.maximum(cost, 1e-9)
        s = np.maximum(sigma, 1e-9)
        return s / c

    eps = 1e-12
    s = np.maximum(sigma, eps)
    c = np.maximum(cost, eps)

    # Estimate number of evaluations made so far.
    # 5 initial uniform points, each with cost 1 + mean(x_scaled).
    # For uniform on [0,1]^2, mean(x_scaled) ≈ 0.5, so cost per point ≈ 1.5.
    # Total cost after 5 points ≈ 7.5.
    # Each subsequent point adds ~1.5 cost on average.
    total_cost = np.sum(cost)
    n_initial = 5
    avg_cost_per_point = 1.5
    est_n_evals = n_initial + max(0.0, (total_cost - n_initial * avg_cost_per_point)) / avg_cost_per_point
    est_n_evals = max(5.0, min(est_n_evals, 30.0))

    # Budget remaining: total budget is 30.
    budget_remaining = max(0.0, 30.0 - total_cost)
    frac_budget_remaining = budget_remaining / 30.0

    # Adaptive kappa: high when few evaluations (explore), low when many (exploit).
    # Start at ~4.0 and decay to ~0.5 as est_n_evals grows.
    # Use a smooth decay: kappa = kappa_max - (kappa_max - kappa_min) * (est_n_evals / 30.0)
    kappa_max = 4.0
    kappa_min = 0.5
    kappa = kappa_max - (kappa_max - kappa_min) * (est_n_evals / 30.0)
    kappa = np.clip(kappa, kappa_min, kappa_max)

    # Expected Improvement for minimization:
    # EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    # where z = (f_best - mu(x)) / sigma(x)
    # This is the expected amount by which f(x) beats f_best.
    z = (f_best - mu) / s
    # Clip z to avoid numerical issues in norm.cdf/pdf
    z = np.clip(z, -10.0, 10.0)
    ei = (f_best - mu) * norm.cdf(z) + s * norm.pdf(z)
    # EI can be slightly negative due to numerical error; clip to 0
    ei = np.maximum(ei, 0.0)

    # Cost penalty: prefer cheaper evaluations.
    # Scale the penalty by how much budget remains: when budget is tight,
    # cost matters more.
    # cost_penalty = cost_weight * cost / (budget_remaining + 1.0)
    # When budget_remaining is large, penalty is small; when small, penalty is large.
    cost_weight = 0.3 + 0.7 * (1.0 - frac_budget_remaining)
    cost_penalty = cost_weight * c

    # Final score: EI - cost_penalty
    # Higher score = more likely to evaluate
    score = ei - cost_penalty

    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    return score
# EVOLVE-BLOCK-END
