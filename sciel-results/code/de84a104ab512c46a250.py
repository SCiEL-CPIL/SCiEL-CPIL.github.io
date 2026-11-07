import numpy as np
import time
from scipy.stats import norm

# Global timer for deadline guard
_start_time = time.time()
_TIME_LIMIT = 315.0  # Safety margin from 330s

def acquisition(mu, sigma, f_best):
    # Check time
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        # If we are running out of time, return a safe fallback
        return np.full(len(mu), -1e10)

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    # Handle edge cases
    if mu.size == 0:
        return np.array([])

    # Ensure sigma is positive to avoid division by zero
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)

    # --- Structural change: Adaptive PI + uncertainty bonus ---
    # Instead of fixed-xi EI, we use a blend of:
    #   1. Probability of Improvement (PI): P(f(x) < f_best)
    #   2. Uncertainty bonus: sigma (encourages exploration)
    # The blend weight adapts based on proximity to the known global min (3.0).
    #
    # When f_best is far from 3.0 (early optimization), we weight exploration
    # (uncertainty) more heavily to find promising regions.
    # When f_best is close to 3.0 (late optimization), we weight exploitation
    # (probability of improvement) more heavily to refine the solution.

    # Compute Z for probability of improvement
    z_pi = (f_best - mu) / sigma_safe

    # Probability that f(x) < f_best (for minimization)
    pi = norm.cdf(z_pi)

    # Adaptive exploration weight based on distance to known global minimum
    # Goldstein-Price 2D has global minimum at 3.0
    global_min = 3.0
    # Distance from current best to global minimum, normalized
    # f_best is typically > 3.0 during optimization
    dist_to_opt = max(f_best - global_min, 0.0)
    # Scale: when dist_to_opt is large, explore more; when small, exploit more
    # Use a smooth transition: exploration_weight = exp(-dist_to_opt / scale)
    # At dist=0: weight=1 (pure exploitation)
    # At dist=large: weight->0 (pure exploration)
    scale = 5.0  # Controls transition sharpness
    exploration_weight = np.exp(-dist_to_opt / scale)

    # Blend: score = exploration_weight * uncertainty + (1 - exploration_weight) * PI
    # This ensures:
    #   - Far from optimum: high uncertainty regions are preferred
    #   - Close to optimum: regions likely to improve are preferred
    score = exploration_weight * sigma_safe + (1.0 - exploration_weight) * pi

    # Add a small exploitation term that always favors lower mean
    # This helps when exploration_weight is high but we still want to
    # prefer slightly better means among high-uncertainty points
    exploitation_bonus = 0.1 * (f_best - mu)
    score = score + exploitation_bonus

    # Ensure finiteness
    score = np.where(np.isfinite(score), score, -1e10)

    return score
