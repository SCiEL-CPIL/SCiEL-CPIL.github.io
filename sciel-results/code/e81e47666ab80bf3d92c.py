"""Cost-aware acquisition: Probability-weighted improvement with adaptive exploration."""
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = 325.0  # seconds, safety margin before 330s kill


def _remaining_time():
    return _DEADLINE - (time.time() - _START_TIME)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using probability-weighted improvement
    with adaptive exploration for Bayesian optimization of cosine8 8d.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviations
    f_best : float - best observed value so far
    cost : array of shape (2000,) - evaluation costs
    
    Returns:
    --------
    array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Deadline guard
    if _remaining_time() < 3.0:
        sigma_safe = np.maximum(sigma, 1e-10)
        cost_safe = np.maximum(cost, 1e-10)
        z = (f_best - mu) / sigma_safe
        z = np.clip(z, -15, 15)
        # Simple EI/cost fallback
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        scores = ei / cost_safe
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)

    # z-score for improvement calculations
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -15, 15)

    # Core signal: Probability-weighted improvement
    # P(mu < f_best) * (f_best - mu) / cost
    # This directly maximizes expected regret reduction per unit cost
    # and inherently down-weights high-cost candidates with marginal improvement
    phi_z = norm.cdf(z)
    phi_weighted_improvement = phi_z * (f_best - mu)
    # Ensure non-negative (only count potential improvements)
    phi_weighted_improvement = np.maximum(phi_weighted_improvement, 0.0)
    
    # Cost-normalized core signal
    core_score = phi_weighted_improvement / cost_safe

    # Exploration component: reward high uncertainty in regions that could
    # plausibly beat f_best. Use a soft-gated uncertainty score.
    # Gate by the probability of improvement so exploration is concentrated
    # where improvement is plausible.
    exploration = sigma_safe * phi_z / cost_safe

    # Adaptive weighting: more exploration early (when f_best is far from min)
    # cosine8 minimum is -0.8, typical initial f_best ~ 0.5-1.0
    min_possible = -0.8
    # progress: 1 when f_best is at minimum, 0 when at worst possible (~1.0)
    progress = (f_best - min_possible) / (1.0 - min_possible + 1e-10)
    progress = np.clip(progress, 0.0, 1.0)

    # Exploration weight: higher when we haven't found the minimum yet
    # Use a power to make it more aggressive early
    w_explore = 0.20 + 0.30 * (progress ** 1.5)
    w_explore = np.clip(w_explore, 0.15, 0.50)

    # Normalize both components to comparable scales using z-score normalization
    n = len(core_score)
    
    # Normalize core_score
    core_mean = np.mean(core_score)
    core_std = np.std(core_score)
    if core_std < 1e-12:
        core_std = 1e-12
    core_norm = (core_score - core_mean) / core_std

    # Normalize exploration
    exp_mean = np.mean(exploration)
    exp_std = np.std(exploration)
    if exp_std < 1e-12:
        exp_std = 1e-12
    exp_norm = (exploration - exp_mean) / exp_std

    # Clip normalized scores to avoid extreme outliers
    core_norm = np.clip(core_norm, -5, 5)
    exp_norm = np.clip(exp_norm, -5, 5)

    # Combine: primarily exploitation via probability-weighted improvement,
    # with meaningful exploration
    combined = (1.0 - w_explore) * core_norm + w_explore * exp_norm

    # Ensure finite
    combined = np.where(np.isfinite(combined), combined, 0.0)

    return combined
