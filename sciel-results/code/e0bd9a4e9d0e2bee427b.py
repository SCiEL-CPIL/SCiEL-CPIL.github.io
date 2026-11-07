"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = time.time()
TIME_LIMIT = 325.0  # Safety margin before 330s kill


def _time_exceeded():
    return (time.time() - _start_time) > TIME_LIMIT


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Thompson Sampling with adaptive UCB.
    
    Uses a hybrid approach:
    1. Thompson Sampling: draws from the posterior to naturally balance exploration/exploitation
    2. Adaptive UCB: adds a deterministic exploration term that decreases as f_best improves
    
    The Thompson Sampling component provides stochastic diversity, while the adaptive UCB
    ensures consistent exploitation as we converge to the minimum.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed objective value (lower is better).
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-10)
    
    # Check if we're running out of time
    if _time_exceeded():
        # Fallback to a simple EI-based score
        z = (f_best - mu) / sigma_safe
        z_clipped = np.clip(z, -10.0, 10.0)
        ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
        ei = np.maximum(ei, 0.0)
        score = ei / cost_safe
        score = np.where(np.isfinite(score), score, 0.0)
        return np.maximum(score, 0.0)
    
    # --- Thompson Sampling Component ---
    # Draw a single sample from the posterior for each candidate
    # This gives us a stochastic estimate of the objective value
    rng = np.random.default_rng()
    Z = rng.standard_normal(len(mu))
    ts_sample = mu + sigma_safe * Z
    
    # The Thompson Sampling score: minimize the sampled value per unit cost
    # Higher score = better (we want low objective value / low cost)
    ts_score = -ts_sample / cost_safe
    
    # --- Adaptive UCB Component ---
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10.0, 10.0)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive exploration weight: larger when f_best is large (early), smaller when f_best is small (late)
    # For levy 2d, minimum is 0, so f_best starts positive and decreases
    f_best_abs = abs(f_best)
    
    # Beta adapts smoothly: when f_best_abs is large, explore more; when small, exploit more
    # Use a form that gives beta ~ 2.0 when f_best is large, and ~ 0.5 when f_best is small
    beta_base = 2.0
    beta_exploit = 0.5
    scale = 1.0
    # Smooth transition using a logistic-like function
    beta = beta_exploit + (beta_base - beta_exploit) / (1.0 + f_best_abs / scale)
    
    # UCB component: beta * sigma - mu (higher when uncertain or when mean is low)
    ucb = beta * sigma_safe - mu
    ucb = np.maximum(ucb, 0.0)
    
    # --- Combine Components ---
    # Normalize both components to have similar scales
    # Use the magnitude of f_best as a reference scale
    ei_scale = max(f_best_abs, 1e-3)
    
    # Normalize EI and UCB
    ei_normalized = ei / ei_scale
    ucb_normalized = ucb / ei_scale
    
    # Normalize Thompson Sampling score
    # ts_score is negative (since we negate the objective), so we work with -ts_score for normalization
    ts_abs = np.abs(ts_score)
    ts_max = np.max(ts_abs) if np.max(ts_abs) > 1e-10 else 1.0
    ts_normalized = ts_score / ts_max  # Keep the sign: negative values are good
    
    # Weighted combination:
    # - Thompson Sampling provides stochastic exploration
    # - EI provides exploitation signal
    # - UCB provides additional exploration
    w_ts = 0.4
    w_ei = 0.35
    w_ucb = 0.25
    
    # Note: ts_normalized is negative (good candidates have low sampled value),
    # while ei_normalized and ucb_normalized are positive.
    # We want high total score = good candidate.
    # So we flip the sign of ts_normalized to make "good" = positive.
    combined = w_ts * (-ts_normalized) + w_ei * ei_normalized + w_ucb * ucb_normalized
    
    # Cost-aware: divide by cost
    score = combined / cost_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
