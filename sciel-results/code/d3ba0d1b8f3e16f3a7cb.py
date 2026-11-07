<<<<<<< SEARCH
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Expected improvement per unit cost.
    z = (f_best - mu) / (sigma + 1e-9)
    return ((f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)) / cost
=======
import numpy as np
from scipy.special import ndtr, exp
import time

_start_time = time.time()
_TIME_LIMIT = 290.0

def acquisition(mu, sigma, f_best, cost):
    current_time = time.time()
    if current_time - _start_time > _TIME_LIMIT:
        # Greedy exploitation: pick the point with the lowest mean
        # to maximize the chance of finding a better minimum in the remaining time.
        # Return -mu so that the lowest mu gets the highest score.
        return -mu
    
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement using scipy.special for better performance/stability
    cdf_val = ndtr(z)
    pdf_val = exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    
    ei = (f_best - mu) * cdf_val + sigma_safe * pdf_val
    
    # Cost-aware: EI per unit cost
    return ei / cost
>>>>>>> REPLACE
