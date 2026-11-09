"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_START_TIME = time.time()
DEADLINE_MARGIN = 10.0 # Seconds before the 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Args:
        mu: Array of shape (N,) with GP mean predictions.
        sigma: Array of shape (N,) with GP standard deviations.
        f_best: The lowest observed objective value.
        cost: Array of shape (N,) with evaluation costs.
        
    Returns:
        Array of shape (N,) with acquisition scores. Higher is better.
    """
    # Check deadline
    # If we are close to the limit, use a fast, deterministic fallback
    if time.time() - _START_TIME > 330.0 - DEADLINE_MARGIN:
        # Fallback: Maximize -mu / cost (exploitation with cost awareness)
        # Add small epsilon to cost to avoid div by zero (though cost >= 1)
        safe_cost = np.maximum(cost, 1e-6)
        return -mu / safe_cost

    # Thompson Sampling
    # Sample from the posterior: y ~ N(mu, sigma^2)
    # We want to minimize y, so we maximize -y / cost
    # Note: sigma can be very small or zero. np.random.normal handles scale=0 by returning loc.
    # However, to be numerically safe and ensure exploration if sigma is tiny but not zero, 
    # we just use the provided sigma.
    
    # Generate random normal samples
    # sigma is standard deviation
    samples = mu + sigma * np.random.randn(len(mu))
    
    # Calculate score: We want to find the minimum.
    # A low sample value is good. A low cost is good.
    # Score = - sample / cost
    # This rewards low samples (negative of small positive = small negative? No.)
    # If sample is 0.1, cost 1.5 -> -0.066
    # If sample is 10.0, cost 1.5 -> -6.66
    # We want to MAXIMIZE the score.
    # -0.066 > -6.66. So lower samples get higher scores. Correct.
    
    safe_cost = np.maximum(cost, 1e-6)
    scores = -samples / safe_cost
    
    # Ensure finite scores
    # If samples or cost are inf/nan, replace with -inf or 0?
    # If a sample is inf, score is -inf. Good.
    # If cost is 0 (impossible here), score is inf.
    # We need to return finite scores.
    scores = np.nan_to_num(scores, nan=-1e6, posinf=-1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
