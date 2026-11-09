import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Probability of Improvement (PI)
    with a dynamic uncertainty bonus, normalized via min-max for robustness.
    """
    t_start = time.time()
    
    # Ensure inputs are properly shaped and finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Dynamic uncertainty bonus:
    # Normalize sigma to [0, 1] relative to the max sigma in the batch.
    # This ensures the exploration term scales with the current level of uncertainty
    # in the GP, decaying as the model becomes more confident (sigma decreases).
    max_sigma = np.max(sigma_safe)
    if max_sigma > 1e-10:
        sigma_norm = sigma_safe / max_sigma
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Structural change: Use PI as the primary driver, with a small weight on uncertainty.
    # This prioritizes cheap, high-probability improvements in early stages.
    # The 0.1 factor is a structural choice to keep exploration secondary to exploitation
    # but present enough to avoid getting stuck in local minima of the GP mean.
    raw_score = pi + 0.1 * sigma_norm
    
    # Min-Max normalization for robustness against scale changes in the raw score.
    # This ensures the relative ranking is preserved and the magnitude is bounded.
    min_raw = np.min(raw_score)
    max_raw = np.max(raw_score)
    if max_raw - min_raw > 1e-10:
        score_norm = (raw_score - min_raw) / (max_raw - min_raw)
    else:
        score_norm = np.zeros_like(raw_score)
    
    # Cost-aware adjustment: divide by cost to prefer cheaper evaluations
    scores = score_norm / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time - if we're getting close to the limit, return immediately
    t_elapsed = time.time() - t_start
    if t_elapsed > 280:
        return scores
    
    return scores
