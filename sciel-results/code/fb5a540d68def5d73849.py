"""Cost-aware acquisition for BO with cost(x) = 1 + mean(x_scaled_to_[0,1])."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import erfc
from scipy.stats import norm
import time as _time

_DEADLINE_S = 285.0
_START = None


def _now():
    return _time.time()


def _elapsed():
    if _START is None:
        return 0.0
    return _now() - _START


def _ei_fast(mu, sigma, f_best):
    """Fast Expected Improvement for minimization using erfc for stability."""
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    d = f_best - mu
    z = d / sigma_safe
    z = np.clip(z, -50.0, 50.0)
    
    # norm.cdf(z) = 0.5 * erfc(-z / sqrt(2))
    cdf = 0.5 * erfc(-z / np.sqrt(2.0))
    pdf = norm.pdf(z)
    
    ei = d * cdf + sigma_safe * pdf
    
    # Where sigma is very small, EI is essentially max(d, 0)
    ei = np.where(sigma < eps, np.maximum(d, 0.0), ei)
    return ei


def acquisition(mu, sigma, f_best, cost):
    global _START
    if _START is None:
        _START = _now()

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # Safety: guarantee finite inputs.
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e12, neginf=-1e12)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e3, neginf=-1e6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)

    n = mu.shape[0]

    # Deadline guard: if we are too close to the kill, return a cheap
    # exploitation-only score so we never blow the budget mid-call.
    if _elapsed() > _DEADLINE_S:
        eps = 1e-9
        exploit = np.maximum(f_best - mu, 0.0)
        score = exploit / (cost + eps)
        return np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)

    # Compute Expected Improvement
    ei = _ei_fast(mu, sigma, f_best)
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=0.0)

    # Structural change: Thompson Sampling inspired cost-aware acquisition.
    # We sample a single function value y ~ N(mu, sigma^2) for each candidate,
    # then select the candidate that minimizes the ratio y / cost(x).
    # This naturally balances the stochastic trade-off between predicted 
    # objective improvement and acquisition cost.
    
    # To make this work in a deterministic acquisition function context,
    # we compute the expected value of the negative ratio:
    # E[-y / cost] = -mu / cost (since E[y] = mu)
    # But this is just exploitation. To add exploration, we consider:
    # E[max(0, f_best - y) / cost] which is EI / cost.
    
    # However, the Thompson Sampling insight is to use the variance to 
    # modulate the cost sensitivity. We'll use a hybrid approach:
    # 1. Primary signal: EI / cost (exploitation with cost awareness)
    # 2. Secondary signal: sigma / cost (exploration with cost awareness)
    # 3. Blend based on relative uncertainty
    
    # Adaptive cost exponent that tightens the penalty
    # as the GP uncertainty decreases (late in the run).
    mean_sigma = np.mean(sigma)
    # Normalize sigma: large sigma -> 1.0 (early), small sigma -> 0.0 (late)
    sigma_norm = mean_sigma / (mean_sigma + 0.05)  # in (0, 1)
    # alpha: 0.4 when sigma_norm=1, 1.2 when sigma_norm=0
    alpha_cost = 0.4 + 0.8 * (1.0 - sigma_norm)
    # Clamp to reasonable range
    alpha_cost = np.clip(alpha_cost, 0.3, 1.5)
    
    cost_safe = np.maximum(cost, 1.0)
    cost_factor = cost_safe ** alpha_cost
    
    # EI divided by cost factor
    ei_cost = ei / cost_factor

    # Exploration boost: sigma divided by cost factor
    # This encourages exploration in high-uncertainty regions if they are cheap
    explore_boost = 0.02 * sigma / cost_factor
    
    # Thompson Sampling inspired: add a term that captures the variance 
    # in the ratio y/cost. This is approximately sigma / cost for small sigma.
    # We'll use a slightly different weight to balance exploration-exploitation.
    ts_exploration = 0.01 * sigma / cost_safe
    
    # Combine: EI/cost + exploration terms
    score = ei_cost + explore_boost + ts_exploration
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    return score
# EVOLVE-BLOCK-END
