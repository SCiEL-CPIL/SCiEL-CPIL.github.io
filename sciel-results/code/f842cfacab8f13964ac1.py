"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Pre-compute constants for speed
_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0  # Safety margin before 330s kill


def _norm_cdf(x):
    """Fast, stable normal CDF using erf."""
    return 0.5 * (1.0 + np.erf(x / np.sqrt(2.0)))


def _norm_pdf(x):
    """Fast, stable normal PDF."""
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) and a
    noise-adaptive exploration term. We use a hybrid that favors
    exploitation when the posterior is confident and exploration when
    uncertain.
    
    We use a weighted combination:
      score = EI + kappa * sigma
    
    where kappa is chosen to balance exploration. We also add a small
    penalty for very high uncertainty to avoid pathological picks, and
    ensure all outputs are finite.
    """
    # Handle numerical edge cases
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Standardized distance
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * _norm_cdf(z) + sigma_safe * _norm_pdf(z)
    
    # Exploration term: reward high uncertainty, but dampen extreme values
    # Use a soft cap to avoid runaway exploration
    sigma_capped = np.minimum(sigma_safe, 0.5)
    exploration = 0.15 * sigma_capped
    
    # Combine
    score = ei + exploration
    
    # Ensure finite outputs
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
