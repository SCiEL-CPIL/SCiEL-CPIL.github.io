import numpy as np
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Epsilon to prevent division by zero
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Posterior sampling-based acquisition
    # Draw latent function samples from GP posterior and estimate expected improvement
    n_samples = 50
    
    # Generate random normal samples for each candidate point
    rng = np.random.default_rng(42)
    samples = rng.standard_normal((n_samples, len(mu)))
    
    # Scale by sigma and add mu
    # f_samples[i, j] = mu[j] + sigma[j] * samples[i, j]
    f_samples = mu[None, :] + sigma_safe[None, :] * samples
    
    # For each sample, compute improvement over f_best
    # improvement = max(0, f_best - f_samples)
    improvements = np.maximum(0, f_best - f_samples)
    
    # Average improvement across samples
    scores = np.mean(improvements, axis=0)
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check deadline
    if time.time() - start_time > 320:
         pass
         
    return scores
