"""Cost-aware acquisition for rastrigin-2d BO; combined approach."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Hard wall-clock deadline guard
_DEADLINE_SECONDS = 300.0
_START_TIME = time.time()


def _safe_return(scores, fallback):
    """Ensure scores are finite and shape (2000,)."""
    if scores is None:
        return fallback
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (2000,):
        return fallback
    if not np.all(np.isfinite(scores)):
        scores = np.where(np.isfinite(scores), scores, -1e18)
    return scores


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition combining exploitation (EI) and exploration (UCB),
    with adaptive weighting based on f_best proximity to the minimum (0).
    """
    # Hard deadline guard: return a simple fallback if we're running out of time
    if time.time() - _START_TIME > _DEADLINE_SECONDS - 5.0:
        fallback = -mu / (cost + 1e-9)
        return _safe_return(fallback, np.zeros(2000, dtype=np.float64))

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma and cost are positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)

    # --- Exploitation: Expected Improvement (EI) ---
    # For minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # --- Exploration: UCB-like bonus ---
    # For minimization, we want low mu + high uncertainty
    # Score = -mu + kappa * sigma (higher is better)
    # Adaptive kappa: larger when f_best is far from 0 (early), smaller when close (late)
    f_best_clipped = np.clip(f_best, 0.0, 100.0)
    kappa = 3.0 * np.exp(-f_best_clipped / 5.0)
    exploration = -mu + kappa * sigma

    # --- Cost normalization ---
    # Divide by cost to prefer cheaper evaluations
    ei_score = ei / cost
    explore_score = exploration / cost

    # --- Adaptive weighting between EI and exploration ---
    # When f_best is large (early), weight exploration more
    # When f_best is small (late), weight exploitation more
    # Use a smooth sigmoid-like transition
    weight_explore = 0.3 + 0.5 * np.exp(-f_best_clipped / 3.0)
    weight_exploit = 1.0 - weight_explore

    # Combine scores
    scores = weight_exploit * ei_score + weight_explore * explore_score

    # Add a small pure uncertainty bonus to help escape local minima
    scores_unc = sigma / cost
    scores = 0.95 * scores + 0.05 * scores_unc

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e18)

    return _safe_return(scores, np.zeros(2000, dtype=np.float64))
# EVOLVE-BLOCK-END
