"""Cost-aware acquisition with log-barrier, sublinear exploration, and adaptive weighting."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = None

def _get_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using log-barrier PI with sublinear exploration.
    
    Core idea:
    - PI (Probability of Improvement) * sigma^gamma / cost^delta
    - log1p(PI) saturates exploitation to prevent over-concentration
    - sigma^0.7 gives sublinear exploration (balanced across uncertainty scales)
    - cost^0.5 softens cost penalty
    - Adaptive blend: when PI is high (confident), weight exploitation more;
      when PI is low (uncertain), weight exploration more.
    """
    _get_start_time()
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 315.0:
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        pi = norm.cdf(z)
        scores = np.log1p(pi) * np.power(sigma_safe, 0.7) / np.power(np.maximum(cost, 1e-8), 0.5)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Probability of Improvement (for minimization)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    pi = np.clip(pi, 1e-10, 1.0 - 1e-10)
    
    # Log-barrier: saturates exploitation smoothly
    log_pi = np.log1p(pi)
    
    # Sublinear exploration: sigma^gamma with gamma < 1
    gamma = 0.7
    sigma_sub = np.power(sigma_safe, gamma)
    
    # Soft cost penalty: cost^delta with delta < 1
    delta = 0.5
    cost_soft = np.power(cost_safe, delta)
    
    # Core score: log1p(PI) * sigma^gamma / cost^delta
    core_score = log_pi * sigma_sub / cost_soft
    
    # Adaptive exploration bonus:
    # When the best PI is low (model is uncertain about improvement), add more exploration
    # When the best PI is high (model is confident), rely more on exploitation
    pi_max = np.max(pi)
    
    # Exploration weight: high when pi_max is low (uncertain), low when pi_max is high (confident)
    # Use a smooth transition
    explore_weight = 0.3 * (1.0 - pi_max) ** 1.5
    
    # Uncertainty-based exploration: normalized sigma
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-12:
        sigma_norm = sigma_safe / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Low-mean bonus: reward points with low predicted mean
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    if mu_max - mu_min > 1e-12:
        mu_norm = (mu - mu_min) / (mu_max - mu_min)
        low_mean_bonus = 1.0 - mu_norm  # 1 = best (lowest mean), 0 = worst
    else:
        low_mean_bonus = np.ones_like(mu)
    
    # Exploration score: combination of uncertainty and low mean
    exploration_score = 0.6 * sigma_norm + 0.4 * low_mean_bonus
    
    # Final score
    scores = core_score + explore_weight * exploration_score
    
    # Normalize to avoid scale issues
    scores_max = np.max(scores)
    if scores_max > 1e-12:
        scores = scores / scores_max
    
    # Safety: ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
