"""Cost-aware acquisition for rosenbrock 2d BO."""
import time
import numpy as np
from scipy.stats import norm

_DEADLINE = None
_MARGIN = 5.0  # seconds of safety margin before 300s limit


def _time_left():
    if _DEADLINE is None:
        return np.inf
    return max(0.0, _DEADLINE - time.time() - _MARGIN)


def acquisition(mu, sigma, f_best, cost):
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 300.0

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    n = mu.shape[0]

    # Guard against degenerate inputs
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Expected improvement (standard EI, lower is better objective)
    delta = f_best - mu
    z = delta / sigma_safe
    ei = delta * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Cost-aware score: EI per unit cost
    score = ei / cost_safe

    # Exploitation boost: reward low predicted mu (direct exploitation)
    # Normalize mu so the boost is scale-free
    mu_range = mu.max() - mu.min()
    if mu_range > 1e-12:
        mu_norm = (mu - mu.min()) / mu_range  # 0 = best predicted, 1 = worst
        exploit = 1.0 - mu_norm
    else:
        exploit = np.ones(n)
    # Blend: 0.85 * cost-aware EI + 0.15 * exploitation
    score = 0.85 * score + 0.15 * exploit * (ei + 1e-6) / (cost_safe + 1e-9)

    # Finite guard
    score = np.where(np.isfinite(score), score, 0.0)

    # --- Local refinement on the acquisition surface for the top candidate ---
    # We don't have x coordinates, but we can refine the *score* by looking at
    # the structure of mu/sigma. Since rosenbrock is smooth, the best EI point
    # is likely near the GP minimum. We do a small local search in the
    # index space is not possible without x. Instead, we apply a small
    # multiplicative boost to points that are locally optimal in score.
    #
    # A simpler and safer structural improvement: apply a small sharpening
    # to the top-k scores to break ties in favor of the most promising.
    if n > 1 and _time_left() > 1.0:
        # Sharpen: amplify the top 1% of scores slightly
        top_k = max(1, int(n * 0.001))
        top_idx = np.argpartition(-score, top_k)[:top_k]
        # Among top-k, boost the single best a bit more
        best_idx = top_idx[np.argmax(score[top_idx])]
        score[best_idx] *= 1.05

    # Final finite guard
    score = np.where(np.isfinite(score), score, 0.0)
    return score
