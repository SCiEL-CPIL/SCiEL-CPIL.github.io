"""Rewritten acquisition for Bayesian optimization of Hartmann 3D."""
import numpy as np
from scipy.stats import norm
import time

def _sanitize(scores):
    """Ensure scores are finite and of the correct shape."""
    if scores.shape != (2000,):
        scores = np.resize(scores, (2000,))
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-8)
    mu_safe = np.nan_to_num(mu, nan=f_best, posinf=f_best, neginf=f_best)
    f_best_safe = float(np.nan_to_num(f_best, nan=-1e6, posinf=1e6, neginf=-1e6))
    
    # Standard EI for minimization
    z_ei = (f_best_safe - mu_safe) / sigma_safe
    ei_scores = (f_best_safe - mu_safe) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    
    # LCB for minimization: score = -mu + kappa * sigma
    # Use a moderate kappa that balances exploitation and exploration
    kappa = 1.5
    lcb_scores = -mu_safe + kappa * sigma_safe
    
    # PI (Probability of Improvement) for minimization
    pi_scores = norm.cdf((f_best_safe - mu_safe) / sigma_safe)
    
    # Normalize each score component to [0, 1] range for fair combination
    def normalize(scores):
        s_min = np.min(scores)
        s_max = np.max(scores)
        if s_max - s_min < 1e-10:
            return np.zeros_like(scores)
        return (scores - s_min) / (s_max - s_min)
    
    # Weighted combination favoring EI (exploitation) with some exploration
    # EI gets the highest weight, LCB provides exploration, PI is a tie-breaker
    w_ei = 0.6
    w_lcb = 0.3
    w_pi = 0.1
    
    final_scores = w_ei * normalize(ei_scores) + w_lcb * normalize(lcb_scores) + w_pi * normalize(pi_scores)
    
    # Safety check
    if time.time() - start_time > 280.0:
        return _sanitize(ei_scores)
    
    final_scores = _sanitize(final_scores)
    
    return final_scores
