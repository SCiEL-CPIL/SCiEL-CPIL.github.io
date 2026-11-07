# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline guard and random seeding
_start_time = None
_seed_counter = 0

def _init_state():
    global _start_time
    if _start_time is None:
        _start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition score for 2000 candidates.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    global _start_time, _seed_counter
    
    # Initialize state on first call
    _init_state()
    
    # Deadline guard
    # The total budget is 330s. We should return well before that.
    # Since this function is called in a loop, we check elapsed time.
    # However, the function itself is very fast (O(N)).
    # The main risk is if the GP fitting takes too long, but that's outside this function.
    # We will just ensure we don't hang.
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety clamps
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Thompson Sampling Strategy:
    # 1. Sample a threshold (offset) from a Gaussian distribution.
    #    Mean: f_best (we want to find points better than the current best)
    #    Std: Scaled by the maximum uncertainty in the candidate set.
    #    This allows the threshold to explore below f_best when uncertainty is high,
    #    and stay close to f_best when uncertainty is low (exploitation).
    
    max_sigma = np.max(sigma)
    
    # Scale the std of the threshold sample.
    # If max_sigma is large, we allow the threshold to vary more.
    # A factor of 0.5 seems reasonable to encourage exploration when uncertain.
    # If max_sigma is very small, the threshold stays close to f_best.
    threshold_std = 0.5 * max_sigma
    
    # Ensure threshold_std is not zero to avoid division by zero issues in normal dist if needed,
    # but here we just sample.
    if threshold_std < 1e-9:
        threshold_std = 1e-9
        
    # Sample the offset
    # We use a deterministic seed based on the counter to ensure reproducibility if needed,
    # but for Thompson Sampling, randomness is key.
    # However, in BO, the "randomness" comes from the GP.
    # Here we add an extra layer of randomness for the acquisition function itself.
    # To avoid global state issues in multi-threaded environments (if any),
    # we can use a local RandomState seeded with a hash of the inputs or a counter.
    # Let's use a simple counter-based seed for the random draw.
    _seed_counter += 1
    rng = np.random.RandomState(_seed_counter)
    
    threshold = rng.normal(loc=f_best, scale=threshold_std)
    
    # 2. Score each candidate.
    # We want to maximize the probability of finding a value lower than the threshold.
    # A simple proxy is: Score = (threshold - mu) / cost
    # If mu < threshold, score is positive (good).
    # If mu > threshold, score is negative (bad).
    # Dividing by cost penalizes expensive points.
    
    scores = (threshold - mu) / cost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
