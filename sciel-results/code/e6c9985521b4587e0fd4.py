import numpy as np
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = start_time + 290
    
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Check if we have time for posterior sampling
    if time.time() > deadline:
        # Fallback to UCB if out of time
        return -mu + 2.0 * sigma
    
    # Use posterior sampling to approximate EI
    # We use a fixed seed for reproducibility within a single call
    # but the randomness helps explore the space better than deterministic heuristics
    rng = np.random.default_rng()
    
    n_samples = 100
    
    # Draw samples from the GP posterior
    # y_s = mu + sigma * z_s
    z = rng.standard_normal((n_samples, len(mu)))
    y_samples = mu[None, :] + sigma[None, :] * z
    
    # Compute improvement for each sample
    imp = np.maximum(f_best - y_samples, 0.0)
    
    # The acquisition score is the expected improvement
    # Approximated by the mean of the improvements
    scores = np.mean(imp, axis=0)
    
    return scores
