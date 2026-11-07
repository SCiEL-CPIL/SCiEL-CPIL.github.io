import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    
    # If we are close to the 300s limit, return a safe score
    # The prompt says "return the best result found with a safety margin"
    # This is tricky because we don't have the "best result" (x), only scores.
    # If we are about to timeout, the current evaluation might be the last one.
    # We should probably just return the standard score, as it's fast.
    # The timeout risk is likely in the GP update, not the acquisition.
    # But if we are being paranoid:
    if elapsed > 295:
        # Return a score that favors the current best?
        # We don't have x.
        # Just return a constant?
        # If we return 0, it might pick index 0.
        # Let's just return the PI score. It's fast.
        pass

    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    cost_safe = np.maximum(cost, 1e-9)
    score = pi / cost_safe
    
    return score
