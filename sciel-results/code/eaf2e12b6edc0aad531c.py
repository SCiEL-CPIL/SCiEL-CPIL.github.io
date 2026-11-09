import numpy as np
from scipy.stats import norm
import time

# Global state
_start_time = None
_step_counter = 0
_last_f_best = None
# We use a fixed base seed for the RNG to ensure the sequence of samples is deterministic
# and reproducible for the same step count, which helps in consistent evaluation.
# However, since we don't know the exact step count reliably across seeds (as the 
# evaluator resets state per seed but our global state might persist if not careful),
# we rely on the step counter to advance the RNG state.
_base_seed = 42
_rng = np.random.default_rng(_base_seed)

def acquisition(mu, sigma, f_best):
    global _start_time, _step_counter, _last_f_best, _rng
    
    if _start_time is None:
        _start_time = time.time()
    
    # Detect new seed/run
    # If f_best is significantly higher than the last observed min, assume new run
    # The function values are in the range roughly [-3, 0]
    if _last_f_best is not None and f_best > _last_f_best + 0.5:
        _step_counter = 0
        # Reset RNG for the new run to ensure consistent behavior per seed
        _rng = np.random.default_rng(_base_seed)
    else:
        _step_counter += 1
        
    _last_f_best = f_best
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 320:
        # Fallback to greedy (min mu) if something goes wrong or time is running out
        # This is a safety measure. TS is very fast.
        return -mu

    # Thompson Sampling:
    # Sample from the posterior: y = mu + sigma * z, where z ~ N(0, 1)
    # We want to minimize f, so we pick the candidate with the lowest sampled value.
    # The acquisition score returned should be such that the MAX score is chosen.
    # Therefore, we return -sampled_value.
    
    # Generate random samples
    # sigma is standard deviation.
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    z = _rng.standard_normal(len(mu))
    
    # Sampled objective values
    y = mu + sigma_safe * z
    
    # We want the evaluator to pick the point with the highest score.
    # Since we want to minimize y, the best point is argmin(y).
    # So the score for that point should be the highest.
    # score = -y achieves this.
    scores = -y
    
    return scores
