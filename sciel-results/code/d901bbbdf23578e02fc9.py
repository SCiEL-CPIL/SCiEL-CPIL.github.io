import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE_MARGIN = 15.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling with Improvement Reward acquisition function.
    
    This is a structural change from Expected Improvement (EI).
    It samples from the GP posterior to create stochastic exploration,
    rewarding points where the sampled value is better than f_best,
    normalized by the evaluation cost.
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 330.0 - _DEADLINE_MARGIN:
        # Fallback to a simple, fast heuristic if time is running out
        # Using EI as a fallback is robust
        mu_f = np.asarray(mu, dtype=np.float64)
        sigma_f = np.asarray(sigma, dtype=np.float64)
        cost_f = np.asarray(cost, dtype=np.float64)
        mu_f = np.where(np.isfinite(mu_f), mu_f, 0.0)
        sigma_f = np.where(np.isfinite(sigma_f), sigma_f, 0.0)
        cost_f = np.where(np.isfinite(cost_f) & (cost_f > 0), cost_f, 1.0)
        safe_sigma = np.maximum(sigma_f, 1e-10)
        Z = (f_best - mu_f) / safe_sigma
        EI = (f_best - mu_f) * norm.cdf(Z) + sigma_f * norm.pdf(Z)
        EI = np.maximum(EI, 0.0)
        score = EI / cost_f
        return np.where(np.isfinite(score), score, 0.0)
    
    # Convert to numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Safe sigma for sampling
    # If sigma is 0, the sample is just mu.
    # We don't need to clip sigma for multiplication, but for stability:
    safe_sigma = np.maximum(sigma, 1e-10)
    
    # Generate standard normal samples
    # Using np.random.randn is fast
    z = np.random.randn(len(mu))
    
    # Sample from the posterior: y ~ N(mu, sigma^2)
    y_sample = mu + safe_sigma * z
    
    # Compute improvement over current best
    # We are minimizing, so improvement is f_best - y_sample
    # Clip to 0 to avoid negative scores (no reward for worse values)
    improvement = np.maximum(0.0, f_best - y_sample)
    
    # Cost-aware score
    # Higher score is better
    score = improvement / cost
    
    # Final safety check
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
