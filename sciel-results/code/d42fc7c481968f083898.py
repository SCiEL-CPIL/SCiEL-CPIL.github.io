"""Rewritten acquisition for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Cache for precomputed constants to speed up repeated calls
_CDF_CACHE = {}
_PDF_CACHE = {}
_DEADLINE = None  # Set once per run if needed, but we'll just use a simple check


def _norm_cdf(x):
    """Vectorized norm CDF with basic caching for common values (optional optimization)."""
    return norm.cdf(x)


def _norm_pdf(x):
    """Vectorized norm PDF."""
    return norm.pdf(x)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function that combines Expected Improvement (EI) with
    a UCB-like exploration term to balance exploitation and exploration.
    
    mu: array of shape (2000,) - GP mean predictions
    sigma: array of shape (2000,) - GP standard deviation predictions
    f_best: float - best (lowest) observed function value
    
    Returns: array of shape (2000,) - scores, higher is better for selection
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-score for EI
    # Since we're minimizing, improvement is f_best - mu (positive when mu < f_best)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * _norm_cdf(z) + sigma_safe * _norm_pdf(z)
    
    # UCB term for exploration - balances exploitation and exploration
    # The beta parameter controls the trade-off; we use a moderate value
    beta = 2.0
    ucb = -mu + beta * sigma_safe
    
    # Combine EI and UCB: use EI as the primary signal, add a small UCB boost
    # This helps escape local optima in the search
    # We normalize both to have comparable scales
    ei_max = np.max(ei) if np.any(ei > 0) else 1.0
    ucb_max = np.max(ucb) if len(ucb) > 0 else 1.0
    
    # Weighted combination: 70% EI, 30% UCB
    score = 0.7 * (ei / (ei_max + 1e-9)) + 0.3 * (ucb / (ucb_max + 1e-9))
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
