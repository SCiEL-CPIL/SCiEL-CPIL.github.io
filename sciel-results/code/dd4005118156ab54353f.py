"""Cost-aware acquisition for BO with cost(x) = 1 + mean(x_scaled_to_[0,1])."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time as _time

_DEADLINE_S = 245.0
_START = None


def _now():
    return _time.time()


def _elapsed():
    if _START is None:
        return 0.0
    return _now() - _START


def acquisition(mu, sigma, f_best, cost):
    global _START
    if _START is None:
        _START = _now()

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # Safety: guarantee finite inputs.
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e12, neginf=-1e12)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e3, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)

    n = mu.shape[0]

    # Deadline guard: if we are too close to the kill, return a cheap
    # exploitation-only score so we never blow the budget mid-call.
    if _elapsed() > _DEADLINE_S:
        eps = 1e-9
        exploit = np.maximum(f_best - mu, 0.0)
        score = exploit / (cost + eps)
        return np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)

    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)

    # --- Expected Improvement (EI) for minimization ---
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10.0, 10.0)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.where(sigma > 0, ei, np.maximum(f_best - mu, 0.0))
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=0.0)

    # --- Cost-aware scaling ---
    # cost is in [1, 2].
    # Use a power law for tunable cost sensitivity.
    cost_safe = np.maximum(cost, 1.0)
    alpha_cost = 0.8
    cost_factor = cost_safe ** alpha_cost

    ei_aware = ei / cost_factor

    # --- Confidence-adaptive exploitation ---
    # When sigma is tiny, the model is confident; exploit more.
    sigma_ref = 0.5
    logit = np.log(sigma_safe / sigma_ref)
    conf = 1.0 / (1.0 + np.exp(logit))
    conf = np.clip(conf, 0.0, 1.0)

    # Exploitation score: predicted improvement, cost-aware
    exploit = np.maximum(f_best - mu, 0.0) / cost_factor

    # Blend weights:
    # - Base: 100% Gaussian EI (simplified from Student-t blend for stability)
    # - Add exploitation when confident
    exploit_blend = 0.15 * conf

    # Final score: (1 - exploit_blend) * ei_aware + exploit_blend * exploit
    score = (1.0 - exploit_blend) * ei_aware + exploit_blend * exploit

    # Ensure finiteness.
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    return score
# EVOLVE-BLOCK-END
