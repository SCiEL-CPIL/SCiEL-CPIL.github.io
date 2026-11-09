import numpy as np
import time

# Global variable to track start time for the deadline guard
_start_time = None

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for Bayesian Optimization.
    
    Strategy:
    Instead of using a deterministic confidence bound (like LCB/UCB) which relies on a 
    hyperparameter beta to balance exploration and exploitation, we use Thompson Sampling.
    We draw a sample from the posterior distribution of the objective function at each 
    candidate point and select the point with the lowest sampled value (for minimization).
    
    This approach is materially different from LCB because:
    1. It is stochastic, providing a natural exploration mechanism without tuning beta.
    2. It adapts the exploration/exploitation balance automatically based on the 
       posterior uncertainty and mean.
    3. It is often more effective in multi-modal landscapes like Hartmann 6d by 
       allowing larger jumps to unexplored regions when uncertainty is high.
    """
    global _start_time
    
    # Initialize start time if not set
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard: The evaluator kills after 330s. 
    # We should return a safe, deterministic result if we are running out of time.
    # However, this function is called many times. The "kill" is on the total wall-clock.
    # We can't really "return the best result found" because we don't have the history 
    # of evaluated points, only the current mu/sigma.
    # If we are near the deadline, we should probably fall back to a very fast, 
    # deterministic heuristic to ensure we don't crash or timeout in this specific call,
    # although the timeout is on the whole loop.
    # A safe fallback is to return a score based on -mu (pure exploitation) or a 
    # simple LCB, which is very fast.
    
    elapsed = time.time() - _start_time
    # If we have used more than 320 seconds, we are in danger.
    # The remaining 10 seconds might not be enough for the GP update and evaluation 
    # in the caller's loop. 
    # If we detect high elapsed time, we return a simple, fast score.
    if elapsed > 320:
        # Fallback: Exploitation only (minimize mu)
        # Return -mu so that the lowest mu gets the highest score.
        mu_arr = np.asarray(mu, dtype=np.float64)
        return -mu_arr

    # Standard Thompson Sampling
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Draw samples from the posterior
    # samples = mu + sigma * N(0, 1)
    # We want to minimize the objective, so we want the lowest sample.
    # The interface requires "higher score = evaluate next".
    # So we return -sample.
    samples = mu + sigma * np.random.randn(len(mu))
    
    # Return negative samples so that the argmax corresponds to the argmin of the samples
    scores = -samples
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
