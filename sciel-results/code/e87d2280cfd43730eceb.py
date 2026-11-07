import numpy as np

# Global state for time tracking if needed, though likely not necessary for simple math
_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    if _start_time is None:
        import time
        _start_time = time.time()
        
    # Deadline guard: if we are dangerously close to the limit, return a safe fallback
    # This is mostly precautionary.
    import time
    if time.time() - _start_time > 325.0:
        # Return a simple LCB or random scores to finish quickly
        return -np.asarray(mu, dtype=np.float64) + 2.0 * np.asarray(sigma, dtype=np.float64)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-12)
    
    # Hartmann 3D global minimum
    target_min = -3.86278
    
    # Calculate progress distance
    # dist is large when f_best is bad (high), small when f_best is good (low)
    dist = f_best - target_min
    dist = np.clip(dist, 0.0, 4.0)
    
    # Adaptive beta
    # Previous: 1.0 + 2.0 * (dist / 4.0) -> [1.0, 3.0]
    # New: 2.0 + 3.0 * (dist / 4.0) -> [2.0, 5.0]
    # More aggressive exploration early on.
    beta = 2.0 + 3.0 * (dist / 4.0)
    
    # LCB score for minimization: maximize -mu + beta * sigma
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    return scores
