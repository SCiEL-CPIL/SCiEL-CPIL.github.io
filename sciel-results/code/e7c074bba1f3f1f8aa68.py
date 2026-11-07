"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = _start_time + 300  # 300s limit, keep a margin

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed objective value so far.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
        We return -sampled_values so that the point with the lowest sampled value 
        has the highest score.
    """
    # Check deadline
    if time.time() > _DEADLINE - 10:
         # Fallback: return UCB with low exploration to be safe and fast
         return -mu + 1.0 * sigma

    # Standard Thompson Sampling: draw independent samples for each point
    # This allows for diverse exploration across the candidate set
    z = np.random.normal(0, 1, size=mu.shape)
    
    # The sampled value for each point is mu + sigma * z_i
    # Since we are minimizing, we want to select the point with the lowest sampled value.
    # The acquisition function returns scores where higher is better.
    # Therefore, we return the negative of the sampled values.
    sampled_values = mu + sigma * z
    return -sampled_values
# EVOLVE-BLOCK-END
