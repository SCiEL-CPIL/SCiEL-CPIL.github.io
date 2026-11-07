"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds before the 330s kill

def _within_deadline():
    return (time.time() - _start_time) < (330.0 - _DEADLINE_MARGIN)


def acquisition(mu, sigma, f_best, cost):
    """Max(EI, LCB) acquisition with cost-awareness and deadline guard."""
    if not _within_deadline():
        # Fallback: pure exploitation (pick lowest mu / cost)
        return -mu / (cost + 1e-9)
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Lower Confidence Bound (exploration)
    # Beta controls exploration strength; for Griewank (smooth, low-dim),
    # moderate exploration is appropriate
    beta = 0.5
    lcb = mu + beta * sigma_safe  # lower bound on true value (we want to minimize)
    
    # Cost normalization: divide by cost to get per-unit-cost value
    ei_cost = ei / (cost + eps)
    lcb_cost = -lcb / (cost + eps)  # negate because lower LCB = better
    
    # Max operation: pick the better of exploitation or exploration
    # This prevents additive interference
    score = np.maximum(ei_cost, lcb_cost)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
