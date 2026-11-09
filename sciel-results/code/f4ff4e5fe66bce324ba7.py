"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def _safe_std(s):
    """Replace zeros/negatives with a tiny positive value to avoid division by zero."""
    s = np.asarray(s, dtype=np.float64)
    return np.where(s > 1e-12, s, 1e-12)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: a blend of Expected Improvement (EI) and
    Upper Confidence Bound (UCB), with an adaptive weight that shifts
    from exploration (UCB-heavy) in early iterations to exploitation
    (EI-heavy) later.

    Parameters
    ----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points.
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best (lowest) observed value so far.

    Returns
    -------
    scores : ndarray, shape (2000,)
        Higher score means evaluate next.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 300s kill

    mu = np.asarray(mu, dtype=np.float64)
    sigma = _safe_std(sigma)

    # Compute EI
    z_ei = (f_best - mu) / sigma
    # Clip z to avoid numerical issues in cdf/pdf for extreme values
    z_ei = np.clip(z_ei, -50.0, 50.0)
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma * norm.pdf(z_ei)
    # EI can be negative if mu > f_best; clip to 0 for scoring
    ei = np.maximum(ei, 0.0)

    # Compute UCB: mu + beta * sigma
    # Use a beta that balances exploration
    beta = 2.0
    ucb = mu + beta * sigma

    # Normalize EI and UCB to [0, 1] range for blending
    # Use min-max normalization with safety for constant arrays
    def _normalize(x):
        mn = np.min(x)
        mx = np.max(x)
        if mx - mn < 1e-15:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    ei_norm = _normalize(ei)
    ucb_norm = _normalize(-ucb)  # negate because lower UCB value is better for minimization, but we want higher score

    # Adaptive weight: use the spread of sigma as a proxy for how much
    # exploration is still needed. When sigma is large relative to the
    # scale of the problem, we should explore more.
    # A simple heuristic: weight on EI increases as we find better values.
    # Since we don't know the iteration count, use the magnitude of f_best
    # relative to the typical scale. For Weierstrass on [-0.5, 0.5],
    # f_best starts around 0.5-1.0 and decreases toward 0.
    # Use a sigmoid-like transition: when f_best is high (early), explore more.
    # f_best ~ 0.5 => w_ei ~ 0.3 (more UCB)
    # f_best ~ 0.05 => w_ei ~ 0.8 (more EI)
    # f_best ~ 0.0 => w_ei ~ 0.95

    # Scale: Weierstrass values are roughly in [0, ~1.2]
    # Use a logistic function centered around 0.1
    scale = 0.1
    w_ei = 1.0 / (1.0 + np.exp((f_best - 0.1) / scale))
    # Clip to reasonable range
    w_ei = np.clip(w_ei, 0.1, 0.95)
    w_ucb = 1.0 - w_ei

    scores = w_ei * ei_norm + w_ucb * ucb_norm

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Deadline check: if we're running out of time, return immediately
    if time.time() > deadline - 5.0:
        return scores

    return scores
# EVOLVE-BLOCK-END
