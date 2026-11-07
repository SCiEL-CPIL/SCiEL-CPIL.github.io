"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    Strategy:
    Instead of computing a deterministic score like EI or UCB, we sample a
    function value from the posterior distribution at each candidate point.
    Specifically, we draw a single sample from a Normal distribution with
    mean mu and standard deviation sigma for each of the 2000 candidates.
    The candidate with the lowest sampled value is selected for evaluation.
    
    This approach (Thompson Sampling) is known to provide a better
    exploration-exploitation trade-off than greedy methods, especially in
    multimodal landscapes like Goldstein-Price, by occasionally selecting
    points with high uncertainty if the sampled value happens to be low.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Posterior mean predictions at candidate points
    sigma : array of shape (2000,)
        Posterior standard deviation predictions at candidate points
    f_best : float
        Best (lowest) observed function value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores, higher is better. 
        We return the negative of the sampled values so that the 
        argmax corresponds to the argmin of the sampled function values.
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-10)
    sigma = np.nan_to_num(sigma, nan=1e-10, posinf=1e10, neginf=1e-10)
    
    # Thompson Sampling: Draw one sample from the posterior for each candidate
    # We use the standard normal random number generator for efficiency
    # samples = mu + sigma * z
    z = np.random.standard_normal(mu.shape)
    samples = mu + sigma * z
    
    # The score should be such that the highest score corresponds to the 
    # most promising candidate (lowest function value).
    # Therefore, we negate the samples.
    scores = -samples
    
    # Ensure all values are finite as per contract
    # Replace NaNs with a very low score (bad) and infinities with large finite values
    scores = np.nan_to_num(scores, nan=-1e10, posinf=1e10, neginf=-1e10)
    
    # Clip to ensure finiteness within reasonable bounds
    max_float = 1e10
    scores = np.clip(scores, -max_float, max_float)
    
    # Deadline guard: If we are running out of time, we could potentially 
    # return a deterministic fallback, but Thompson sampling is O(N) and very fast.
    # The main cost is in the GP evaluation outside this function.
    elapsed = time.time() - start_time
    if elapsed > 250: # Safety margin
         # Fallback to pure exploitation if somehow slow (unlikely)
         scores = -mu
         scores = np.nan_to_num(scores, nan=-1e10, posinf=1e10, neginf=-1e10)
         scores = np.clip(scores, -1e10, 1e10)

    return scores
# EVOLVE-BLOCK-END
