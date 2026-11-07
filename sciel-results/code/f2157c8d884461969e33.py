# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Instead of computing a deterministic score like EI or LCB, we sample a function
    from the posterior GP: f_sample(x) ~ N(mu(x), sigma(x)^2).
    The acquisition score is the negative of this sample.
    Maximizing this score is equivalent to minimizing the sampled function value.
    
    This approach has strong theoretical guarantees and often outperforms EI in
    multi-modal landscapes like Hartmann 3D because it explores regions with high
    uncertainty (high sigma) that *could* contain the global minimum, rather than
    just focusing on the expected improvement which can be conservative.
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety check for time (330s limit, we are fast, but just in case)
    if time.time() - start_time > 325:
         # Fallback to something very fast if we somehow stalled
         return -mu

    # Thompson Sampling:
    # Draw a random sample from the standard normal distribution
    # The sampled function value at each candidate point is:
    # y_sample = mu + sigma * z
    # where z ~ N(0, 1)
    
    # We want to minimize the function, so the "score" for selection (which we maximize)
    # is -y_sample.
    # However, the contract says "higher meaning evaluate next".
    # So we return - (mu + sigma * z).
    
    # To ensure robustness and avoid potential numerical issues with extremely large
    # or small values (though unlikely with standardized targets), we just compute directly.
    
    # Generate random noise
    z = np.random.normal(0, 1, size=mu.shape)
    
    # Compute sampled values
    sampled_values = mu + sigma * z
    
    # The acquisition score is the negative of the sampled value
    # (since we want to minimize the function, the best candidate has the lowest sampled value,
    #  so we assign it the highest score)
    scores = -sampled_values
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
