"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timer for deadline guard
_start_time = time.time()
_DEADLINE = 295.0  # Safety margin from 330s kill


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition with adaptive kappa.
    
    For minimization: EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    Higher EI means better candidate.
    
    We add a small exploration bonus based on sigma to avoid getting stuck.
    
    mu: array of shape (N,) - posterior mean in objective units
    sigma: array of shape (N,) - posterior std in objective units
    f_best: scalar - best (lowest) observed value so far
    """
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        return np.zeros_like(mu)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Numerical stability
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Standard EI for minimization
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF of standard normal, phi is PDF
    
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -10, 10)
    
    # Compute EI
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Add a small exploration bonus: kappa * sigma
    # This helps escape local minima
    # When f_best is large, we want more exploration
    # When f_best is small, we want less exploration
    kappa = 0.1 + 0.5 / (1.0 + f_best / 10.0)
    
    scores = ei + kappa * sigma_safe
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
