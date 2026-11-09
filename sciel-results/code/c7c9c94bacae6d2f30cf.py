import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    Uses Thompson Sampling: sample a function value from the posterior
    N(mu, sigma^2) and return the negative of the sample.
    Higher scores correspond to lower predicted objective values.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 1.0
    
    # Thompson Sampling: Draw a sample from the posterior
    # f_sample ~ N(mu, sigma^2)
    # We use a fixed seed for reproducibility within a run? 
    # No, random sampling is the point of TS.
    # However, to avoid pathological cases where we always pick the same point
    # due to low variance, we rely on the natural randomness.
    
    # Generate standard normal noise
    noise = np.random.randn(mu.shape)
    
    # Sample from posterior
    f_sample = mu + sigma * noise
    
    # We want to minimize the objective.
    # The evaluator picks the argmax of the returned scores.
    # So we return -f_sample.
    # If f_sample is low (good), -f_sample is high (good score).
    scores = -f_sample
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    return scores
