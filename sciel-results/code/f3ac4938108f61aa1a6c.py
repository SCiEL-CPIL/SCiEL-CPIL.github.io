"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


# Deadline guard: keep explicit wall-clock tracking
_START_TIME = time.time()
_TIME_LIMIT = 300.0  # evaluator kills at 330s, leave margin
_TIME_MARGIN = 10.0  # safety margin seconds


def _time_left():
    return _TIME_LIMIT - _TIME_MARGIN - (time.time() - _START_TIME)


def _logsumexp(a, axis=None):
    """Numerically stable log-sum-exp."""
    a_max = np.max(a, axis=axis, keepdims=True)
    # Handle case where all values are -inf
    a_max_safe = np.where(np.isfinite(a_max), a_max, 0.0)
    out = np.log(np.sum(np.exp(a - a_max_safe), axis=axis, keepdims=True)) + a_max_safe
    if axis is not None:
        out = np.squeeze(out, axis=axis)
    return out


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Branin 2D.
    
    Uses a smooth combination of Expected Improvement (EI) and UCB-style
    exploration, with the exploration weight adapted based on GP uncertainty.
    Uses log-sum-exp for smooth max over beta values.
    
    Parameters:
    -----------
    mu : ndarray, shape (n,)
        GP mean at candidate points (objective units)
    sigma : ndarray, shape (n,)
        GP standard deviation at candidate points (objective units)
    f_best : float
        Best observed value so far
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores (higher = more promising)
    """
    # Check time budget - if we're running out, return a simple fallback
    if _time_left() < 1.0:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute UCB scores for several beta values
    # UCB score = -mu + beta * sigma (for minimization, higher is better)
    betas = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
    n = len(mu)
    
    # Shape: (len(betas), n)
    ucb_scores = -mu[None, :] + betas[:, None] * sigma_safe[None, :]
    
    # Normalize each beta's UCB scores to [0, 1] range
    ucb_scores_norm = np.zeros_like(ucb_scores)
    for i in range(len(betas)):
        row = ucb_scores[i, :]
        row_max = np.max(row)
        row_min = np.min(row)
        if row_max - row_min > 1e-12:
            ucb_scores_norm[i, :] = (row - row_min) / (row_max - row_min)
        else:
            ucb_scores_norm[i, :] = 0.5
    
    # Use log-sum-exp for a smooth max over beta values
    # This gives a soft-max that favors points that are good across multiple betas
    # Temperature: lower = closer to hard max, higher = more uniform
    temperature = 2.0
    log_weights = np.log(np.full(len(betas), 1.0))  # uniform weights
    # Smooth max: log(sum(exp((score - max)/T))) * T + max
    ucb_max_per_point = np.max(ucb_scores_norm, axis=0)
    shifted = (ucb_scores_norm - ucb_max_per_point[None, :]) / temperature
    log_sum_exp_shifted = _logsumexp(shifted, axis=0)
    ucb_smooth_max = ucb_max_per_point + temperature * log_sum_exp_shifted
    
    # Adapt exploration weight based on GP uncertainty
    # When sigma is high relative to the range of mu, we should explore more
    mu_range = np.max(mu) - np.min(mu)
    if mu_range < 1e-12:
        mu_range = 1.0
    
    # Mean sigma relative to mu range
    mean_sigma_rel = np.mean(sigma_safe) / mu_range
    
    # Exploration weight: higher when uncertainty is high
    # Base weight 0.3, up to 0.5 when very uncertain
    w_explore = 0.3 + 0.2 * min(mean_sigma_rel / 0.5, 1.0)
    w_exploit = 1.0 - w_explore
    
    # Scale UCB to EI's scale
    ei_max = np.max(ei)
    if ei_max < 1e-12:
        ei_max = 1e-9
    scale = ei_max
    
    # Final score
    score = w_exploit * ei + w_explore * ucb_smooth_max * scale
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
