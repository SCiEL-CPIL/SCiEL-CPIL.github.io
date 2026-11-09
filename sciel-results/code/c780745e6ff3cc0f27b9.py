import numpy as np
import time

# Global state for time management and Thompson Sampling randomness
_start_time = None
_ts_state = None
_ts_seed_counter = 0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Styblinski-Tang 1D.
    
    Strategy: Thompson Sampling.
    We draw a single sample from the Gaussian posterior N(mu, sigma^2) for each candidate.
    The candidate with the lowest sampled value is selected for evaluation.
    Since the interface requires higher scores to indicate better candidates,
    we return -sample.
    
    Parameters:
    mu: np.ndarray of shape (2000,), posterior mean in objective units.
    sigma: np.ndarray of shape (2000,), posterior standard deviation in objective units.
    f_best: float, the lowest observed value so far.
    
    Returns:
    scores: np.ndarray of shape (2000,), higher scores indicate better candidates to evaluate next.
    """
    global _start_time
    global _ts_state
    global _ts_seed_counter
    
    if _start_time is None:
        _start_time = time.time()
        
    # Time budget guard
    if time.time() - _start_time > 320:
        # Fallback to exploitation if we are running out of time
        return -mu

    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-10)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-10)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Initialize or update the random state for Thompson Sampling
    # We use a global RandomState to ensure reproducibility within the run if needed,
    # but since the task doesn't specify seed control for the acquisition itself
    # beyond the evaluator's seeds, we just use a persistent state.
    # Using a fresh seed based on a counter to avoid correlation with previous calls
    # is not strictly necessary if we just advance the state, but let's keep it simple.
    
    if _ts_state is None:
        # Use a fixed base seed for the acquisition's internal randomness to be deterministic
        # relative to the function call sequence, or just use the system RNG.
        # The evaluator runs seeds 0, 1, 2. We want the exploration to be stochastic.
        # Using np.random.RandomState() without a seed uses system entropy.
        _ts_state = np.random.RandomState()
        
    # Draw samples: x ~ N(mu, sigma)
    # For each candidate i, sample z_i ~ N(0, 1)
    # sample_i = mu_i + sigma_i * z_i
    # We want to minimize sample_i.
    # Score = -sample_i.
    
    z = _ts_state.standard_normal(size=mu.shape)
    samples = mu + sigma * z
    
    # The score should be higher for better candidates (lower function value).
    scores = -samples
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
