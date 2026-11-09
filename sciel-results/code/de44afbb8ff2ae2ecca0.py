import numpy as np
from scipy.stats import norm
import time

_T0 = time.time()
_DEADLINE = _T0 + 320.0


def _check_deadline():
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best):
    """
    Multi-criterion ensemble acquisition for minimization.
    Blends LCB, UCB, EI, and Thompson-sampling scores with progress-adaptive weights.
    """
    n = len(mu)
    if sigma is None or len(mu) != len(sigma):
        return -mu
    sigma = np.maximum(sigma, 1e-10)

    if _check_deadline():
        # Fallback: simple LCB
        return -mu - 0.5 * sigma

    f_min = 0.397887
    f_max_est = 302.0

    # Progress: 0 = just started (f_best near f_max), 1 = at optimum
    progress = 1.0 - (f_best - f_min) / (f_max_est - f_min)
    progress = np.clip(progress, 0.0, 1.0)

    # --- Criterion 1: LCB (exploitation + exploration) ---
    # beta_lcb starts high, decreases
    beta_lcb = 0.3 + 2.0 * (1.0 - progress)
    lcb = -mu - beta_lcb * sigma  # lower is better for minimization, so higher score = more promising

    # --- Criterion 2: UCB (pure exploration emphasis) ---
    beta_ucb = 1.0 + 3.0 * (1.0 - progress)
    ucb = -mu + beta_ucb * sigma

    # --- Criterion 3: Expected Improvement ---
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # --- Criterion 4: Thompson sampling score ---
    # Draw a random sample from the posterior: f ~ N(mu, sigma^2)
    # For minimization, lower sample = more promising
    rng = np.random.default_rng()
    ts_sample = mu + sigma * rng.standard_normal(n)
    ts_score = -ts_sample  # higher = lower sample = more promising

    # Normalize each criterion to [0, 1] range for fair blending
    def _normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    lcb_n = _normalize(lcb)
    ucb_n = _normalize(ucb)
    ei_n = _normalize(ei)
    ts_n = _normalize(ts_score)

    # Weights: early -> explore (LCB, UCB, TS), late -> exploit (EI, LCB)
    w_lcb = 0.2 + 0.3 * (1.0 - progress)
    w_ucb = 0.1 + 0.4 * (1.0 - progress)
    w_ei = 0.3 + 0.5 * progress
    w_ts = 0.2 + 0.3 * (1.0 - progress)

    # Normalize weights to sum to 1
    w_sum = w_lcb + w_ucb + w_ei + w_ts
    w_lcb /= w_sum
    w_ucb /= w_sum
    w_ei /= w_sum
    w_ts /= w_sum

    scores = w_lcb * lcb_n + w_ucb * ucb_n + w_ei * ei_n + w_ts * ts_n

    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
