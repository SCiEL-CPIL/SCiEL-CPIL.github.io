"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track wall-clock deadline (330s limit, use 310s margin)
_T0 = time.time()
_DEADLINE = 310.0


def _time_ok():
    return (time.time() - _T0) < _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Bayesian optimization of Styblinski-Tang 2D.
    
    mu, sigma, cost: arrays of shape (2000,)
    f_best: scalar, lowest observed value
    Returns: 2000 finite scores (higher = evaluate next)
    """
    # Deadline guard
    if not _time_ok():
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / np.maximum(cost, 1e-6)
        return np.where(np.isfinite(score), score, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Safety: ensure finite and positive
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    sigma_safe = np.maximum(sigma, 1e-8)

    # Infer progress based on f_best
    f_best_val = float(f_best)
    if not np.isfinite(f_best_val):
        f_best_val = 0.0
    # ST-2d: f(0,0)=0, min=-78.3323
    # progress goes from 0 to 1 as f_best goes from 0 to -78.3323
    progress = np.clip(-f_best_val / 78.3323, 0.0, 1.0)

    # Adaptive parameters based on progress
    # Early: explore more, later: exploit more
    # beta for UCB: high early, low late
    beta = 3.0 - 2.5 * progress  # 3.0 early, 0.5 late
    
    # For EI, we can use a temperature-like scaling
    # Early: we want to be more aggressive about improvement
    # Late: we want to be more conservative

    # Compute Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Compute UCB
    ucb = -mu + beta * sigma_safe

    # Cost-aware scoring
    cost_safe = np.maximum(cost, 1e-6)
    
    # Normalize by cost to prefer cheaper evaluations
    # But don't divide by too small a number
    score_ei = ei / cost_safe
    score_ucb = ucb / cost_safe

    # Blend EI and UCB
    # Early: favor UCB for exploration
    # Late: favor EI for exploitation
    w_ucb = 0.5 - 0.3 * progress  # 0.5 early, 0.2 late
    w_ei = 1.0 - w_ucb            # 0.5 early, 0.8 late

    score = w_ei * score_ei + w_ucb * score_ucb

    # Add a small exploration bonus for regions with high uncertainty
    # relative to the scale of the problem
    # This helps escape local optima
    sigma_max = np.max(sigma_safe) if np.max(sigma_safe) > 0 else 1.0
    uncertainty_bonus = 0.1 * (sigma_safe / sigma_max) * (1.0 - progress)
    score = score + uncertainty_bonus / cost_safe

    # Penalize very high cost points (corners of the domain)
    # Cost ranges from 1 (center) to 2 (corners)
    cost_penalty_weight = 0.05
    cost_excess = np.maximum(cost - 1.3, 0.0)
    penalty = cost_penalty_weight * cost_excess ** 2
    score = score - penalty

    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
