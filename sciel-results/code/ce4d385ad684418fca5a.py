"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import time

# Global counter for dynamic acquisition parameter
_t = 0
_start_time = None

def acquisition(mu, sigma, f_best):
    global _t, _start_time
    
    if _start_time is None:
        _start_time = time.time()
    
    _t += 1
    
    # Deadline guard: return a safe fallback if we're getting close to the limit
    # The evaluator kills at 330s, so we want to finish well before that.
    # Since acquisition is called once per evaluation step (25 times total),
    # and each step involves 2000 candidates, the bottleneck is likely the GP
    # update/evaluation in the outer loop, not this function.
    # However, we'll add a guard just in case.
    elapsed = time.time() - _start_time
    if elapsed > 300:  # 30s margin
        # Return a simple fallback: prioritize low mu
        return -mu
    
    # Ensure sigma is positive to avoid division by zero or log issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Dynamic exploration parameter
    # We blend between exploitation (EI-like) and exploration (UCB-like)
    # As we collect more data, we can shift balance
    
    # For the Styblinski-Tang function, which is multimodal,
    # some exploration is important, but exploitation is key for refinement
    
    # Use a dynamic beta that starts high (exploration) and decreases
    # beta_t = sqrt(log(1 + t)) is a common choice
    beta = math.sqrt(math.log(1.0 + _t))
    
    # Cap beta to avoid excessive exploration
    beta = min(beta, 2.0)
    
    # Compute a combined score:
    # -mu gives exploitation (lower is better, so we negate)
    # beta * sigma gives exploration (higher uncertainty is better)
    # We can also add a small EI-like term for smoothness
    
    # Simple UCB-style score
    scores = -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
