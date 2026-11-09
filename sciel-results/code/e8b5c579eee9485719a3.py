"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE = 330.0  # seconds, evaluator kills at 330s

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    Strategy:
    Instead of computing a deterministic score like EI or UCB, we simulate a sample
    from the posterior distribution at each candidate point. For minimization, we
    want to find the point where the sampled value is lowest. 
    Since the function returns scores where "higher means evaluate next", we return
    the negative of the sampled value.
    
    This approach is often superior in multimodal landscapes (like Michalewicz) because
    the variance (sigma) directly scales the random perturbation. High uncertainty
    regions have a higher chance of yielding a very low (good) sampled value, thus
    naturally driving exploration without needing an explicit beta parameter or
    rank normalization.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (n,)
        Mean of the GP posterior at candidate points.
    sigma : np.ndarray, shape (n,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far (lowest, since we're minimizing).
    
    Returns:
    --------
    scores : np.ndarray, shape (n,)
        Acquisition scores; higher means more desirable to evaluate.
    """
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - 5.0:
        # Fallback: exploit the current best mean (lowest mu)
        return -np.asarray(mu, dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure finite inputs
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Generate a single standard normal random vector
    # We use a fixed seed for the random number generation within the acquisition?
    # No, Thompson sampling relies on the stochasticity of the sample to explore.
    # However, the evaluator calls this function repeatedly.
    # If we use a global RNG, the sequence of samples matters.
    # Standard practice is to just draw from the global state.
    
    # Draw sample: x ~ N(mu, sigma^2)
    # To ensure we don't have issues with extreme outliers affecting the "best" choice
    # in a way that is too erratic, we could clip the z-score, but usually raw TS is robust.
    # Let's just draw.
    
    # One optimization: np.random.normal is fast.
    # sample = mu + sigma * np.random.normal(size=mu.shape)
    
    # However, there is a subtle point. If we just return -sample, we are picking the 
    # candidate that *happens* to have the lowest value in this specific draw.
    # This is the standard Thompson Sampling heuristic for acquisition functions 
    # when you must return a score for a batch of candidates and the caller picks the argmax.
    # Note: Strict TS usually picks one point. Here, we score 2000 points, and the 
    # caller picks the max score (min sampled value). This is effectively "Batch Thompson Sampling"
    # where we pick the best single candidate from the posterior samples.
    
    # To avoid potential issues with the global RNG state being shared across seeds 
    # in the evaluator (if the evaluator reuses the same process), it's safer to 
    # rely on the standard global numpy random state which is initialized by the environment.
    
    # Let's add a small epsilon to sigma to prevent zero variance if the GP becomes certain.
    # (Already handled by max(sigma, 1e-9))
    
    samples = mu + sigma * np.random.normal(size=mu.shape)
    
    # We want higher scores to be selected.
    # Minimization: lower sampled value is better.
    # So score = -sampled_value.
    scores = -samples
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
