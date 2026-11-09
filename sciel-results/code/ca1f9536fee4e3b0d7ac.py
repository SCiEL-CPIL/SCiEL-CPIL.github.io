"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timer for deadline guard
_start_time = time.time()
# The evaluator kills after 330s. We use 325s as a safety margin.
# Since this function is called ~25 times, and each call is fast,
# this is mostly a precaution.
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for Bayesian optimization.
    
    Calculates the expected improvement over the current best observed value
    for each candidate point. Higher scores indicate a higher expected reduction
    in the objective function value.
    
    Parameters:
    -----------
    mu : ndarray, shape (n,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (n,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        The lowest observed objective value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        EI scores; higher means evaluate next.
    """
    # Deadline guard: if we are running out of time, fall back to a very fast
    # deterministic exploitation strategy (lowest mean).
    if time.time() - _start_time > _DEADLINE:
        return -mu

    n = len(mu)
    
    # Ensure sigma is positive to avoid division by zero
    # Using a small epsilon for numerical stability
    eps = 1e-6
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate the standardized improvement z
    # z = (f_best - mu) / sigma
    # If mu < f_best, z is positive (improvement is possible)
    # If mu > f_best, z is negative (improvement is unlikely)
    z = (f_best - mu) / sigma_safe
    
    # Calculate CDF and PDF of the standard normal distribution
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # This is always non-negative.
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # For points where sigma was extremely small (clipped to eps),
    # the formula approximates max(0, f_best - mu).
    # Let's explicitly handle the case where original sigma was near 0
    # to ensure we don't get weird numerical artifacts from the eps clipping
    # although for z->inf, cdf->1, pdf->0, so EI->f_best-mu.
    # For z->-inf, cdf->0, pdf->0, so EI->0.
    # The formula is stable enough.
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(ei), ei, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
