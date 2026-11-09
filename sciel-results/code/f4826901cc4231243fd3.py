"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for Thompson Sampling
_ts_state = {
    'rng': None,
    'last_time': 0.0,
    'f_best': 0.0,
    'iteration': 0
}

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Parameters:
    mu: array of shape (2000,), posterior mean
    sigma: array of shape (2000,), posterior standard deviation
    f_best: float, best observed value so far
    
    Returns:
    scores: array of shape (2000,), sampled values from the posterior
    """
    # Initialize RNG if not set
    if _ts_state['rng'] is None:
        # Use a fixed seed for reproducibility across calls within a run if needed, 
        # but typically we want fresh randomness. 
        # However, to ensure the "seed" behavior of the evaluator works, 
        # we should probably initialize with a seed derived from the first call or just use default.
        # The task says "For each seed 0, 1 and 2 the evaluator starts...". 
        # The acquisition function doesn't receive the seed. 
        # Using np.random.default_rng() without seed uses system entropy.
        # To be safe and deterministic per run if possible, we could just use a global RNG.
        _ts_state['rng'] = np.random.default_rng()
        
    # Update state
    _ts_state['f_best'] = f_best
    _ts_state['iteration'] += 1
    
    # Check time limit (300s per evaluation, but we are in the loop. 
    # The prompt says "Each evaluation must finish within 300 s". 
    # The acquisition call itself is just one step. 
    # We should ensure we don't hang, but sampling is O(N).
    # We add a simple guard just in case, though sampling 2000 floats is trivial.
    current_time = time.time()
    # If we've been running too long globally? No, the evaluator kills the whole program.
    # We just need to return valid scores.
    
    # Thompson Sampling: Sample from N(mu, sigma^2)
    # mu and sigma are in objective units.
    # We want to minimize. So we sample a value, and the "score" for selection 
    # is such that higher is better? 
    # Wait, the contract says: "Return 2000 finite scores, higher meaning evaluate next".
    # In minimization, a *lower* objective value is better.
    # If we sample y ~ N(mu, sigma^2), a low y is good.
    # So the "score" that the evaluator uses to pick the "highest score" 
    # should correspond to the "best" candidate.
    # Therefore, we should return -y (negative of the sampled value) 
    # so that the lowest objective value (most negative) gets the highest score.
    
    samples = _ts_state['rng'].normal(mu, sigma)
    
    # Ensure finite
    scores = -samples
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return scores
# EVOLVE-BLOCK-END
