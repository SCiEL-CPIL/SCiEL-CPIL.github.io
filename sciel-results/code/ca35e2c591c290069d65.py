"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Expected Improvement per unit cost, with a small exploration bonus
    to avoid collapsing onto already-explored low-cost regions.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed function value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Ensure sigma is positive for numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard normal CDF/PDF computations
    z = (f_best - mu) / sigma_safe
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Per-unit-cost improvement
    scores = ei / np.maximum(cost, 1e-9)
    
    # Add a small exploration bonus based on uncertainty to avoid
    # over-concentrating on low-cost, low-improvement regions
    # The bonus is scaled by the global uncertainty to be adaptive
    max_sigma = np.max(sigma_safe)
    if max_sigma > 1e-12:
        # Exploration bonus: proportional to sigma / cost, weighted by a small factor
        # This encourages exploring uncertain regions while still being cost-aware
        exploration = 0.05 * sigma_safe / np.maximum(cost, 1e-9)
        scores = scores + exploration
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
