import numpy as np
import time

# Track start time for deadline guard
_start_time = time.time()
_DEADLINE_MARGIN = 10  # seconds to keep before the 330s limit

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling for minimization.
    
    Parameters:
    mu (np.array): Mean of the GP posterior, shape (2000,)
    sigma (np.array): Standard deviation of the GP posterior, shape (2000,)
    f_best (float): Best observed function value so far.
    
    Returns:
    np.array: Scores for the 2000 candidates, shape (2000,). Higher is better.
    """
    # Deadline guard
    if time.time() - _start_time > 330 - _DEADLINE_MARGIN:
        # Fallback: return zeros or a simple heuristic if time is running out
        # In practice, this should never trigger for 2000 points
        return np.zeros_like(mu)
    
    # Ensure sigma is non-negative and safe
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Draw a sample from the posterior predictive distribution
    # For minimization, we want to sample f(x) = mu(x) - sigma(x) * z
    # where z ~ N(0, 1)
    # The evaluator picks the candidate with the HIGHEST score.
    # We want the evaluator to pick the candidate with the LOWEST sampled f(x).
    # Therefore, we return -f_sample = -mu + sigma * z
    
    z = np.random.standard_normal(len(mu))
    scores = -mu + sigma_safe * z
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return scores
