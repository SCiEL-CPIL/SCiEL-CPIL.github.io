"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Track start time for deadline guard
_t0 = time.time()
_DEADLINE = 320.0  # safety margin before 330s kill


def _safe_pdf(z):
    """Numerically stable standard normal pdf."""
    # pdf = exp(-0.5*z^2) / sqrt(2*pi)
    # Use log-space for very large |z| to avoid underflow issues
    z2 = 0.5 * z * z
    # For |z| > 40, pdf is effectively 0
    log_pdf = -z2 - 0.5 * np.log(2.0 * np.pi)
    pdf = np.exp(np.clip(log_pdf, -700.0, 0.0))
    return pdf


def _safe_cdf(z):
    """Numerically stable standard normal CDF using erfc for precision."""
    # cdf(z) = 0.5 * erfc(-z / sqrt(2))
    # erfc is well-behaved for all real arguments
    from math import erfc as _erfc
    # Vectorized erfc via scipy or manual
    # scipy.special.erfc is vectorized
    try:
        from scipy.special import erfc as _vec_erfc
        return 0.5 * _vec_erfc(-z / np.sqrt(2.0))
    except ImportError:
        # Fallback: use norm.cdf for moderate z, manual for extremes
        cdf = norm.cdf(z)
        # For very large positive z, cdf should be ~1
        cdf = np.where(z > 8.0, 1.0, cdf)
        # For very large negative z, cdf should be ~0
        cdf = np.where(z < -8.0, 0.0, cdf)
        return cdf


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    
    Hybrid of Expected Improvement (EI) and UCB exploration.
    EI captures exploitation; UCB adds exploration for multimodal search.
    
    mu and sigma are in objective units (lower is better).
    Returns 2000 finite scores, higher = evaluate next.
    """
    # Deadline guard
    if time.time() - _t0 > _DEADLINE:
        # Fallback: pure UCB with moderate beta
        sigma_safe = np.maximum(sigma, 1e-8)
        return -mu + 1.0 * sigma_safe

    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement for minimization
    # Improvement = f_best - mu (positive when mu < f_best)
    z = (f_best - mu) / sigma_safe
    
    # Use stable cdf and pdf
    cdf_z = _safe_cdf(z)
    pdf_z = _safe_pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # UCB exploration term: -mu + beta * sigma
    beta = 0.7
    ucb_term = -mu + beta * sigma_safe
    
    # Blend: EI primary, UCB secondary for exploration
    score = ei + 0.4 * ucb_term
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
