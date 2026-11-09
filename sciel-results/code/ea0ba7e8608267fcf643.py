import numpy as np
from scipy.stats import norm
import time

# Global start time to track elapsed time for the entire evaluation session
_start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    Computes the Probability of Improvement (POI) acquisition function.
    
    This function implements a structural change from the standard Expected Improvement (EI)
    to Probability of Improvement (POI). POI focuses on the probability that a candidate
    point will yield a better objective value than the current best. This can provide
    a softer exploration pressure compared to methods that heavily weight the magnitude
    of potential improvement or uncertainty, which might be beneficial for escaping
    local optima in multimodal functions like Branin 2D.

    Parameters:
    mu: array of shape (n_candidates,) - posterior mean
    sigma: array of shape (n_candidates,) - posterior standard deviation
    f_best: float - best observed value so far
    
    Returns:
    scores: array of shape (n_candidates,) - POI values (higher is better)
    """
    
    # Deadline guard: Check if we are approaching the 330s limit.
    # Since this function is fast, this is mostly a formality, but included per directive.
    elapsed = time.time() - _start_time
    if elapsed > 325:
        # If we are very close to the deadline, return a safe fallback.
        # In a real scenario, the caller should handle this, but we ensure we return finite values.
        # Returning zeros or a random uniform score is safer than hanging.
        # However, if we return all zeros, the optimizer might pick the first one.
        # Let's just return the POI calculation which is fast enough.
        pass

    # Ensure sigma is positive to avoid division by zero
    # Use a small epsilon to prevent division by zero and numerical instability
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate the standardized improvement z
    # z = (f_best - mu) / sigma
    # Note: Since we are minimizing, "improvement" means f(x) < f_best.
    # The probability that f(x) < f_best given f(x) ~ N(mu, sigma^2) is:
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma
    
    # Probability of Improvement is the CDF of the standard normal distribution at z
    poi = norm.cdf(z)
    
    # Ensure all scores are finite
    poi = np.where(np.isfinite(poi), poi, 0.0)
    
    return poi
