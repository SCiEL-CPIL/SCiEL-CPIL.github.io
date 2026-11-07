import numpy as np
import time

_start_time = time.time()
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best):
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        return -mu.copy()
    
    # Multi-sample Thompson Sampling
    # We sample multiple latent function values from the GP posterior
    # and for each candidate, take the negative of the minimum sampled value
    # This directly optimizes for the minimum potential value at each point
    
    # Number of samples - balance between accuracy and computation
    n_samples = 50
    
    # Sample n_samples standard normal vectors
    Z = np.random.normal(0, 1, size=(n_samples, len(mu)))
    
    # For each sample i and candidate j:
    # f_i(x_j) = mu(x_j) + sigma(x_j) * Z[i, j]
    # We want the minimum over samples for each candidate:
    # min_i f_i(x_j) = min_i (mu(x_j) + sigma(x_j) * Z[i, j])
    #                 = mu(x_j) + sigma(x_j) * min_i Z[i, j]
    # (since sigma > 0, the minimum is achieved by the minimum Z)
    
    # So we just need the minimum of Z across samples for each candidate
    min_Z = np.min(Z, axis=0)
    
    # Score is negative of the minimum sampled value (higher = better)
    scores = -(mu + sigma * min_Z)
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return scores
