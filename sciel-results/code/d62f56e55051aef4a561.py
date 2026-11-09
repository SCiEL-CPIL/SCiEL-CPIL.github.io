import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    
    Uses a structural blend of Thompson Sampling (for exploration diversity)
    and Expected Improvement (for exploitation precision), with adaptive
    weighting based on posterior uncertainty relative to current best.
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0)
    
    # Compute Thompson Sampling sample
    # Use a deterministic seed derived from the state to avoid interfering with evaluator RNG
    # but vary it slightly based on f_best to change exploration pattern as we improve
    sum_mu = np.sum(mu)
    seed = int(abs(sum_mu) * 12345 + abs(f_best) * 7777) % (2**32)
    rng = np.random.default_rng(seed)
    
    # Draw samples from the posterior
    ts_samples = mu + sigma_safe * rng.standard_normal(len(mu))
    
    # For minimization, we want to pick the point with the lowest sample
    # So the acquisition score for TS is -ts_samples
    ts_score = -ts_samples
    
    # Normalize both EI and TS scores to have zero mean and unit variance for blending
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-12
    ei_norm = (ei - ei_mean) / ei_std
    
    ts_mean = np.mean(ts_score)
    ts_std = np.std(ts_score) + 1e-12
    ts_norm = (ts_score - ts_mean) / ts_std
    
    # Adaptive weighting: 
    # When f_best is close to 0 (good), exploit more (higher EI weight)
    # When f_best is far from 0 (bad), explore more (higher TS weight)
    # Also consider the overall uncertainty level
    f_best_clamped = np.clip(f_best, 0, 2)
    
    # Weight for EI: goes from 0.3 (explore) to 0.9 (exploit)
    weight_ei = 0.3 + 0.6 * np.exp(-1.5 * f_best_clamped)
    weight_ts = 1.0 - weight_ei
    
    # Blend
    score = weight_ei * ei_norm + weight_ts * ts_norm
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
