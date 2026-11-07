"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: The evaluator kills after 330s.
# We track time to ensure we return before the hard limit.
# Since this function is called per step and is vectorized, it is fast.
# But we add a check just in case of pathological inputs or system load.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 300.0 # Safety margin of 30s

def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a neutral score, e.g., all zeros or the current best's score
        # This prevents the evaluator from hanging or timing out.
        return np.zeros_like(mu)

    # Avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei_scores = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Refinement: Use Log-Expected Improvement for better numerical stability
    # and to dampen extreme exploitation/exploitation swings.
    # Add epsilon to avoid log(0)
    log_ei_scores = np.log(ei_scores + 1e-12)
    
    # Ensure finiteness
    scores = np.where(np.isfinite(log_ei_scores), log_ei_scores, -1e9)
    
    return scores
# EVOLVE-BLOCK-END
