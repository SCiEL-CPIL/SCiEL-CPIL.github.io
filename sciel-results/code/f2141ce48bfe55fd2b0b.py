"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # 300s limit, 10s safety margin

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    
    EI is a structural improvement over linear UCB as it non-linearly
    balances exploitation (low mu) and exploration (high sigma) based
    on the current best value f_best.
    
    Parameters:
    mu: array of shape (N,)
    sigma: array of shape (N,)
    f_best: scalar, lowest observed value
    
    Returns:
    scores: array of shape (N,), higher values indicate better candidates
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback to a simple heuristic if we are running out of time
        # This should rarely happen as EI is fast
        return -mu + 2.0 * sigma
    
    # Avoid division by zero
    sigma_safe = np.where(sigma > 1e-10, sigma, 1.0)
    
    # Z-score for minimization
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If mu < f_best, z > 0, Phi(z) > 0.5, EI > 0.
    # If mu > f_best, z < 0, Phi(z) < 0.5, EI might be small or negative (but we treat as low score).
    
    # Using scipy.stats.norm
    cdf_vals = norm.cdf(z)
    pdf_vals = norm.pdf(z)
    
    scores = (f_best - mu) * cdf_vals + sigma_safe * pdf_vals
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
