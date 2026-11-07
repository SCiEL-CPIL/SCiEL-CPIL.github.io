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
    # cost is in [1, 2]. Use alpha=1.0 for maximum cost sensitivity
    # to spend budget on as many evaluations as possible.
    cost_safe = np.maximum(cost, 1.0)
    cost_factor = cost_safe  # linear cost penalty

    ei_aware = ei / cost_factor

    # --- Adaptive exploitation blend ---
    # As budget depletes (higher mean cost observed), shift toward exploitation.
    # Use mean cost as a proxy for budget progress.
    mean_cost = np.mean(cost_safe)
    # Progress: 0 at start (mean_cost~1), 1 near end (mean_cost~2)
    budget_progress = np.clip((mean_cost - 1.0) / 1.0, 0.0, 1.0)
    
    # Exploitation strength increases with budget progress
    exploit_strength = 0.1 + 0.35 * budget_progress  # 0.1 to 0.45
    
    # Confidence-based exploitation: when sigma is tiny, exploit more
    sigma_ref = 0.1
    conf = 1.0 / (1.0 + np.exp((sigma_safe - sigma_ref) / (sigma_ref * 0.5)))
    conf = np.clip(conf, 0.0, 1.0)
    
    # Combined exploitation blend
    exploit_blend = exploit_strength * (0.5 + 0.5 * conf)
    exploit_blend = np.clip(exploit_blend, 0.0, 0.6)

    # Exploitation score: predicted improvement, cost-aware
    exploit = np.maximum(f_best - mu, 0.0) / cost_factor

    # Final score
    score = (1.0 - exploit_blend) * ei_aware + exploit_blend * exploit

    # Ensure finiteness.
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    return score
# EVOLVE-BLOCK-END
