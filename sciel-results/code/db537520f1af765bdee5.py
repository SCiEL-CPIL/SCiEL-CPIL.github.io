import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    Parameters:
    mu: array-like, shape (n_candidates,), posterior mean
    sigma: array-like, shape (n_candidates,), posterior standard deviation
    f_best: float, best observed function value so far
    
    Returns:
    scores: array-like, shape (n_candidates,), acquisition scores (higher is better)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Clean up inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    sigma = np.maximum(sigma, 1e-8)
    
    # Thompson Sampling:
    # Draw a sample from the posterior distribution N(mu, sigma^2) for each candidate.
    # For minimization, the "best" candidate is the one with the lowest sample.
    # We return scores such that the argmax corresponds to the argmin of the samples.
    # score = -sample
    
    samples = np.random.normal(loc=mu, scale=sigma)
    
    scores = -samples
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
