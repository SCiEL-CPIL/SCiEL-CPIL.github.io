"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard
_DEADLINE = None
_DEADLINE_MARGIN = 2.0  # seconds before the 330s limit

def _check_deadline():
    """Return True if we should stop searching due to deadline."""
    if _DEADLINE is None:
        return False
    return (time.time() - _DEADLINE) > (330.0 - _DEADLINE_MARGIN)


def acquisition(mu, sigma, f_best):
    """
    Quantile-based acquisition function combining lower confidence bound (LCB)
    with expected improvement (EI) for robust exploration/exploitation balance.
    
    Uses a hybrid: primarily LCB (mu - kappa*sigma) for exploration,
    blended with EI for exploitation when close to the optimum.
    """
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time()
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # LCB component: targets regions with high lower-bound potential
    # Using kappa=1.5 for moderate exploration
    lcb = mu - 1.5 * sigma
    
    # EI component: exploitation when confident
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Normalize EI to be on similar scale as LCB for blending
    # Scale EI by a factor to balance with LCB
    ei_scaled = ei * 0.5
    
    # Blend: use LCB primarily, add EI for exploitation
    # This avoids pure exploitation (EI alone) and pure exploration (LCB alone)
    scores = lcb + ei_scaled
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
