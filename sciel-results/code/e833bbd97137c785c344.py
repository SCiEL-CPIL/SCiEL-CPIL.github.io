import numpy as np
from scipy.stats import norm
import time

# Global time tracking for safety
_start_time = time.time()
_time_limit = 300.0 # Seconds

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we are dangerously close to the limit, return a safe fallback
    if time.time() - _start_time > _time_limit - 5.0:
        return np.zeros_like(mu)
    
    # Safe sigma to avoid division by zero
    safe_sigma = np.maximum(sigma, 1e-9)
    
    # Cost-aware Expected Hypar Improvement (EHI)
    # EHI = E[ (f_best - Y) / (cost + (f_best - Y)) ] where Y ~ N(mu, sigma^2)
    # This explicitly models the trade-off between improvement and cost
    # Higher scores indicate better cost-effectiveness
    
    delta = f_best - mu
    z = delta / safe_sigma
    
    # For EHI, we need E[delta / (cost + delta)]
    # Using the identity: E[delta/(c+delta)] where delta ~ N(delta, sigma^2)
    # This can be approximated using the log-normal expectation or direct integration
    # For numerical stability, use the approximation:
    # E[delta/(c+delta)] ≈ delta/(c+delta) * (1 + sigma^2/(c+delta)^2 * (1 - z*phi(z)/Phi(z)))
    # where phi is standard normal pdf and Phi is standard normal cdf
    
    safe_cost = np.maximum(cost, 1e-9)
    
    # Base improvement ratio
    base_ratio = delta / (safe_cost + delta)
    
    # Handle cases where delta is very negative (mu >> f_best)
    # In these cases, the improvement is negative and we should penalize
    mask_positive = delta > 0
    
    # Compute correction factor for positive improvements
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # The correction accounts for the variance of the improvement
    # E[1/(c+X)] where X ~ N(mu, sigma^2) can be approximated
    # Using the delta method or direct approximation
    
    # For positive delta, use the approximation:
    # E[delta/(c+delta)] ≈ delta/(c+delta) + sigma^2 * delta * (2*delta - (c+delta)) / (c+delta)^4
    # Simplified: add a variance-aware bonus
    
    score = np.zeros_like(mu)
    
    if np.any(mask_positive):
        c = safe_cost[mask_positive]
        d = delta[mask_positive]
        s = safe_sigma[mask_positive]
        
        # Direct computation of E[delta/(c+delta)]
        # Using the fact that for X ~ N(mu, sigma^2):
        # E[delta/(c+delta)] = integral of delta/(c+delta) * N(delta; mu, sigma) d(delta)
        # Approximate with midpoint + variance correction
        
        denom = c + d
        base = d / denom
        
        # Variance correction (second order Taylor expansion)
        # f(x) = x/(c+x), f'(x) = c/(c+x)^2, f''(x) = -2c/(c+x)^3
        # E[f(X)] ≈ f(mu) + 0.5*f''(mu)*sigma^2
        var_correction = -2.0 * c / (denom ** 3) * (s ** 2) * 0.5
        
        score[mask_positive] = base + var_correction
    
    # For negative or zero delta, the expected improvement is negative
    # We want to avoid these, so set a small negative score
    score[~mask_positive] = -0.1
    
    # Handle any numerical issues
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
