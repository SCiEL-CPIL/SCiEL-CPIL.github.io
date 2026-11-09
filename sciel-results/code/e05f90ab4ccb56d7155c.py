"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state
_acq_call_count = 0
_start_time = time.time()
TIME_LIMIT = 300.0  # Safety margin from 330s

def acquisition(mu, sigma, f_best):
    global _acq_call_count, _start_time, TIME_LIMIT
    
    # Time guard
    elapsed = time.time() - _start_time
    if elapsed > TIME_LIMIT:
        # Return pure exploitation (minimize mu) as a safe fallback
        return -mu.copy()
    
    _acq_call_count += 1
    
    # Thompson Sampling: draw a single sample from the posterior predictive
    # This naturally balances exploration and exploitation without manual tuning
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Draw a sample from the Gaussian posterior
    sample = mu - sigma_safe * np.random.standard_normal(len(mu))
    
    # Return the sample as scores (lower is better, so we negate for "higher means evaluate next")
    score = -sample
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
