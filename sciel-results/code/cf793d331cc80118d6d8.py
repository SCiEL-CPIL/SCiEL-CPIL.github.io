"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_START_TIME = time.time()

def acquisition(mu, sigma, f_best):
    """
    Computes the acquisition scores using a hybrid of UCB and Expected Improvement.
    """
    # Deadline guard
    if time.time() - _START_TIME > 320:
        return np.zeros_like(mu)
    
    # Adaptive parameters based on the number of evaluations inferred from f_best
    # Early on (f_best is high), explore more. Later (f_best is low), exploit more.
    # Ackley min is 0.
    # Typical f_best starts around 10-20 and drops to 0.
    
    # Use a fixed but well-tuned beta for UCB, or blend with EI.
    # Let's try a blend: Score = -mu + beta * sigma * (1 + EI_factor)
    # Or simply a dynamic beta.
    
    # Dynamic beta: Start high (2.0) and decay as f_best improves (gets closer to 0).
    # If f_best is close to 0, we are in exploitation mode -> low beta.
    # If f_best is far from 0, we are in exploration mode -> high beta.
    
    # Normalization of f_best for beta decay
    # Assume f_best ranges from ~15 (initial) to 0.
    # Let's map f_best to a decay factor.
    # decay = exp(-f_best / scale) ? No, f_best is decreasing.
    # We want beta to decrease as f_best decreases.
    # beta = beta_max * exp(-f_best / scale) + beta_min
    # If f_best = 10, exp(-10/5) = 0.135.
    # If f_best = 0, exp(0) = 1.0.
    # Wait, if f_best is 0 (perfect), we want low beta? 
    # Actually, if f_best is 0, we've found the minimum. We should stop or just exploit.
    # But if f_best is 10, we haven't found it yet, so explore.
    # So beta should be HIGH when f_best is HIGH.
    # beta = beta_min + (beta_max - beta_min) * exp(-f_best / scale)
    # If f_best = 10, exp(-2) = 0.135 -> beta low? No.
    # We want: f_best high -> explore (high beta). f_best low -> exploit (low beta).
    # So beta should correlate with f_best.
    # beta = beta_min + (beta_max - beta_min) * (f_best / f_ref)
    # Let f_ref = 5.0 (arbitrary scale).
    # If f_best = 10, ratio = 2, beta = beta_max.
    # If f_best = 0, ratio = 0, beta = beta_min.
    
    f_ref = 5.0
    ratio = min(f_best / f_ref, 2.0)
    beta_min = 0.5
    beta_max = 2.5
    beta = beta_min + (beta_max - beta_min) * ratio
    
    # Calculate UCB scores
    scores = -mu + beta * sigma
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
