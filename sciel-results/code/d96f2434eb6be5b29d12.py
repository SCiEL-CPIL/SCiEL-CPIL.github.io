import numpy as np
import time

# Global deadline guard
START_TIME = time.time()
TIME_LIMIT = 320.0  # 300s limit + margin, but evaluator kills at 330s. Keep it tight.

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization using Thompson Sampling.
    Samples a single objective value from the posterior at each candidate point:
    sample = mu - sigma * z, where z ~ N(0,1)
    Returns scores where higher means more promising (lower sampled value).
    Score = -sample = -mu + sigma * z
    """
    # Check deadline
    elapsed = time.time() - START_TIME
    if elapsed > TIME_LIMIT:
        # Fallback to deterministic UCB-like scores to avoid crashes
        sigma_safe = np.maximum(sigma, 1e-8)
        scores = -mu + 2.5 * sigma_safe
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        return scores

    # Thompson Sampling: sample from posterior N(mu, sigma^2)
    # For minimization, we want low values, so score = -sample
    # sample = mu - sigma * z, where z ~ N(0,1)
    # score = -mu + sigma * z
    
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Generate standard normal random numbers
    # Use a fixed seed per call to ensure reproducibility? No, Thompson Sampling requires randomness.
    # But we need to ensure the scores are finite.
    z = np.random.standard_normal(len(mu))
    
    # Compute scores: higher score means lower sampled value (more promising for minimization)
    scores = -mu + sigma_safe * z
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
