import numpy as np
import time

# Track elapsed time for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Beale 2D.
    Uses Thompson Sampling with multiple posterior samples.
    
    Parameters:
    mu (np.ndarray): Mean of the GP posterior, shape (2000,).
    sigma (np.ndarray): Standard deviation of the GP posterior, shape (2000,).
    f_best (float): The lowest observed value so far.
    
    Returns:
    np.ndarray: Scores for each candidate, shape (2000,). Higher is better.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we've used too much time, fall back to simple LCB
    elapsed = time.time() - _start_time
    if elapsed > 280:  # Safety margin before 300s limit
        eps = 1e-9
        sigma_safe = np.maximum(sigma, eps)
        beta = 0.2 * (1.0 + f_best / 10.0)
        return -mu + beta * sigma_safe
    
    n_candidates = len(mu)
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Thompson Sampling: draw K samples from the posterior
    # For each candidate x, f(x) ~ N(mu(x), sigma(x)^2)
    # We want to find x that minimizes f(x), so we sample and take the min
    # across samples for each x, then score = -min_sample_value
    # But we need scores for ALL 2000 candidates, so we use:
    # For each candidate, estimate the expected minimum over K samples
    # This approximates the probability of being the minimizer
    
    # Use a moderate number of samples for stability
    n_samples = 10
    
    # Generate random normal matrix: (n_samples, n_candidates)
    Z = np.random.randn(n_samples, n_candidates)
    
    # Sample values: (n_samples, n_candidates)
    samples = mu[np.newaxis, :] + sigma_safe[np.newaxis, :] * Z
    
    # For Thompson sampling in minimization:
    # A candidate is good if it has low values across samples
    # Use the minimum across samples as a robust score
    # Or use a quantile to balance exploration
    # Take the 25th percentile (lower quantile = more exploitation)
    sample_min = np.percentile(samples, 25, axis=0)
    
    # Score: higher is better for minimization, so negate
    scores = -sample_min
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
