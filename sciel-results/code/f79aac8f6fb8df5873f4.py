import numpy as np
import time

# Module-level variable to track the start time for the deadline guard
_start_time = None

def _get_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a Thompson Sampling proxy.
    
    Parameters:
    mu: array of mean predictions, shape (2000,)
    sigma: array of standard deviations, shape (2000,)
    f_best: scalar, the best observed value so far (minimum)
    
    Returns:
    scores: array of scores, shape (2000,), higher is better
    """
    # Deadline guard: If we are very close to the 330s limit, return a safe, trivial score
    # This is a precaution. The actual computation is O(N) and very fast.
    current_time = time.time()
    elapsed = current_time - _get_start_time()
    if elapsed > 325.0:
        # Return a score that doesn't cause issues, e.g., based on -mu to still prefer lower values
        # But to be safe and fast, just return zeros or a simple heuristic
        # The caller picks the max. If we return zeros, it picks index 0.
        # Better to return something deterministic that doesn't break.
        # However, the contract says "higher meaning evaluate next".
        # If we are out of time, any choice is arbitrary.
        # Let's just return -mu as a fallback, which is fast.
        return -mu

    # Ensure sigma is positive to avoid division by zero or negative issues
    # sigma is a standard deviation, so it should be non-negative.
    # We add a small epsilon to avoid division by zero.
    epsilon = 1e-9
    safe_sigma = np.maximum(sigma, epsilon)
    
    # Thompson Sampling proxy:
    # Maximizes the expected improvement relative to uncertainty.
    # This is equivalent to the posterior mean of the improvement (f_best - f) 
    # if we assume a specific prior, but more directly, it's a heuristic 
    # that balances exploitation (f_best - mu) and exploration (1/sigma).
    # High score -> high potential improvement relative to uncertainty.
    scores = (f_best - mu) / safe_sigma
    
    # Ensure scores are finite
    # If mu is very large or sigma is very small, this could overflow, but 
    # with standardized targets and bounded domain, it should be fine.
    # np.nan_to_num handles any potential NaNs or Infs.
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
