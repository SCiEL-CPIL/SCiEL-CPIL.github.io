"""Bayesian optimization acquisition for 2D RBF mixture with Matérn-5/2 GP."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_T0 = time.time()
_DEADLINE = 310.0
_EPS = 1e-12


def _expected_improvement(mu, sigma, f_best):
    """Compute Expected Improvement for minimization."""
    sigma_safe = np.maximum(sigma, _EPS)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)


def _get_progress(eval_count, total_steps=25):
    """Get progress in [0, 1] over the 25 BO evaluation steps."""
    if eval_count <= 0:
        return 0.0
    return min(eval_count / float(total_steps), 1.0)


def _kriging_believer_score(mu, sigma, progress):
    """
    Kriging-Believer acquisition: score = -q_tau(mu, sigma)
    where q_tau is the tau-quantile of N(mu, sigma^2).
    
    Early: tau small (~0.05) -> explore far tails (high uncertainty regions)
    Late:  tau -> 0.5       -> exploit the mean
    """
    sigma_safe = np.maximum(sigma, _EPS)
    # tau goes from 0.05 (early) to 0.5 (late)
    # Use a smooth sigmoid-like transition
    tau = 0.05 + 0.45 * (1.0 - np.exp(-3.0 * progress))
    # Clip tau to avoid extreme quantiles
    tau = np.clip(tau, 0.01, 0.5)
    # q_tau = mu + sigma * z_{1-tau}  (for minimization, we want low quantiles)
    # z_{1-tau} is the (1-tau) quantile of standard normal
    z_val = norm.ppf(1.0 - tau)
    # The quantile of N(mu, sigma^2) at level tau is: mu + sigma * z_tau
    # But for minimization with KB, we score by -q_tau where q_tau is the
    # tau-quantile. Low tau means we're looking at very low quantiles
    # (optimistic for minimization = exploring).
    # Actually for KB in minimization: score = -q_tau(mu,sigma)
    # where q_tau = mu + sigma * ppf(tau)
    # When tau is small, ppf(tau) is very negative, so q_tau << mu,
    # meaning -q_tau >> -mu, favoring high-uncertainty regions.
    z_tau = norm.ppf(tau)
    q_tau = mu + sigma_safe * z_tau
    return -q_tau


def _soft_ei_scale(ei, top_k=50):
    """
    Compute a smooth scale factor for EI based on top-k EI values.
    Uses log-sum-exp style normalization for stability.
    Returns a scalar scale such that top EI values are O(1).
    """
    if ei.size == 0:
        return 1.0
    # Get top-k EI values
    k = min(top_k, ei.size)
    top_vals = np.partition(ei, -k)[-k:]
    top_vals = top_vals[top_vals > _EPS]
    if top_vals.size == 0:
        return 1.0
    # Use the max of top-k as a robust scale
    scale = np.max(top_vals)
    return max(scale, _EPS)


# Track evaluation count
_eval_count = 0


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of 2D RBF mixture.
    
    Uses a Kriging-Believer quantile-based score blended with a
    properly-scaled Expected Improvement term. The KB score smoothly
    transitions from aggressive exploration (low quantile) to
    exploitation (mean) as progress increases.
    
    Args:
        mu: array of shape (2000,) - posterior mean
        sigma: array of shape (2000,) - posterior standard deviation
        f_best: float - lowest observed value
    
    Returns:
        array of shape (2000,) - scores, higher means evaluate next
    """
    global _eval_count
    
    # Deadline guard: fall back to pure EI
    if time.time() - _T0 > _DEADLINE:
        sigma_safe = np.maximum(sigma, _EPS)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.maximum(ei, 0.0)
    
    # Increment evaluation counter
    _eval_count += 1
    
    sigma_safe = np.maximum(sigma, _EPS)
    progress = _get_progress(_eval_count)
    
    # Kriging-Believer score (primary signal)
    kb_score = _kriging_believer_score(mu, sigma_safe, progress)
    
    # Compute EI (secondary exploitation anchor)
    ei = _expected_improvement(mu, sigma_safe, f_best)
    
    # Scale EI using top-k based scaling for stability
    ei_scale = _soft_ei_scale(ei, top_k=50)
    ei_scaled = ei / ei_scale
    
    # Blend weights: KB dominates, EI provides a small exploitation anchor
    # w_ei increases slightly with progress to anchor near the incumbent later
    w_ei = 0.1 + 0.2 * progress  # 0.1 -> 0.3
    w_kb = 1.0 - w_ei
    
    # Normalize KB score to be O(1) by its own spread
    kb_range = np.max(kb_score) - np.min(kb_score)
    if kb_range > _EPS:
        kb_norm = (kb_score - np.min(kb_score)) / kb_range
    else:
        kb_norm = np.zeros_like(kb_score)
    
    # Blend
    score = w_kb * kb_norm + w_ei * ei_scaled
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
