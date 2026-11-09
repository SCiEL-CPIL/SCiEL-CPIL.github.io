"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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
    deadline = start_time + 280.0  # 300s limit, leave 20s safety margin
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-6)
    mu_safe = np.nan_to_num(mu, nan=f_best, posinf=f_best, neginf=f_best)
    f_best_safe = np.nan_to_num(f_best, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    # Candidate 1: Standard EI (Expected Improvement)
    # For minimization: E[f_best - f(x)]
    z_ei = (f_best_safe - mu_safe) / sigma_safe
    ei_scores = (f_best_safe - mu_safe) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    
    # Candidate 2: LCB (Lower Confidence Bound) for minimization
    # We want to minimize mu - kappa * sigma.
    # Score should be high when mu - kappa * sigma is low.
    # score = -(mu - kappa * sigma) = -mu + kappa * sigma
    # Increased kappa from 1.0 to 2.0 to encourage more exploration
    # and escape local minima in Hartmann 3D.
    kappa = 2.0
    lcb_scores = -mu_safe + kappa * sigma_safe
    
    # Hybrid Strategy:
    # The previous version used 0.7 * EI + 0.3 * LCB(kappa=1.0).
    # We increase the weight of the LCB component to 0.4 and its internal
    # exploration coefficient (kappa) to 2.0. This should help the optimizer
    # explore more aggressively while still being guided by the expected improvement.
    w_ei = 0.6
    w_lcb = 0.4
    
    final_scores = w_ei * ei_scores + w_lcb * lcb_scores
    
    # Check time (should be negligible, but safe to include)
    if time.time() - start_time > deadline - start_time:
        return _sanitize(ei_scores)
        
    # Ensure finite
    final_scores = _sanitize(final_scores)
    
    return final_scores
# EVOLVE-BLOCK-END
