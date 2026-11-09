import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling for minimization.
    
    Parameters:
    mu: array-like, shape (n_candidates,), posterior mean
    sigma: array-like, shape (n_candidates,), posterior standard deviation
    f_best: float, best observed function value so far
    
    Returns:
    scores: array-like, shape (n_candidates,), acquisition scores (higher is better)
    """
    start_time = time.time()
    
    # Convert inputs to numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Clean up inputs to ensure numerical stability
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    sigma = np.maximum(sigma, 1e-8)
    
    # Thompson Sampling: Draw a single sample from the posterior for each candidate
    # f(x) ~ N(mu, sigma^2)
    # For minimization, we want low values.
    # The score should be higher for better (lower) values.
    # So we return the negative of the sample.
    
    # Generate standard normal random numbers
    z = np.random.randn(mu.shape[0])
    
    # Sample from the posterior
    samples = mu + sigma * z
    
    # For minimization, higher score means lower sampled value
    scores = -samples
    
    # Handle numerical issues
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 300:
        # This shouldn't happen for Thompson Sampling, but just in case
        return scores
    
    return scores
