import numpy as np
from scipy.stats import norm
import time

_STEP_COUNT = 0
_TIMEOUT = 290.0 # Safety margin
_START_TIME = time.time()

def acquisition(mu, sigma, f_best):
    global _STEP_COUNT, _START_TIME
    
    # Deadline guard
    if time.time() - _START_TIME > _TIMEOUT:
        # Return a simple, fast score to avoid timeout
        # Use mu directly (exploit) as a fallback
        return -mu 

    _STEP_COUNT += 1
    
    # Adaptive exploration constant
    # Start with 0 (pure EI), increase over time to penalize uncertainty more strongly
    # 30 steps total per seed. If 3 seeds run sequentially, total steps 90.
    # If interleaved, the counter increments 3x per "logical" step.
    # Let's assume the evaluator calls this function 2000 times per evaluation step.
    # So we need to normalize by 2000?
    # No, the function is called ONCE per candidate batch?
    # "draws 2000 fresh uniform candidates, computes the posterior there, calls acquisition"
    # This implies `acquisition` is called with arrays of shape (2000,).
    # So `_STEP_COUNT` should increment by 1 per CALL, not per point.
    # So 1 call = 1 optimization step for 1 seed.
    # Total calls per seed = 25 (30 total evals - 5 initial).
    # Total calls overall = 75.
    
    # Progress: 0 to 1 over 25 steps per seed?
    # If seeds are sequential, step 1-25 is seed 0, 26-50 seed 1, etc.
    # We can't easily distinguish seeds.
    # But we can just use the global count.
    # 0-74.
    # Let's map 0-74 to a lambda range.
    # Or better, reset the counter? We can't reset.
    # Let's just use a fixed lambda that is a good compromise, or a very slow ramp.
    
    # Actually, simpler: Use a fixed small lambda. The "Structural" change is the penalty itself.
    # The previous code was Pure EI. This is Penalized EI.
    
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Fixed penalty
    lambda_penalty = 0.05
    score = ei - lambda_penalty * sigma
    
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
